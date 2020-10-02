import logging
import time
import copy
import datetime
import random
import argparse
import pathlib
import wandb
from random import choice
from string import ascii_uppercase
from collections import defaultdict

import torch
import numpy as np
from tensorboardX import SummaryWriter

from src.helpers.torch_metrics import compute_ssim, compute_psnr
from src.helpers.utils import (add_mask_params, save_json, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters, str2bool, str2none)
from src.helpers.data_loading import create_data_loader
from src.reconstruction_model.reconstruction_model_utils import load_recon_model
from src.policy_model.policy_model_utils import (build_policy_model, load_policy_model, build_optim, save_policy_model,
                                                 compute_scores, create_data_range_dict,
                                                 compute_next_step_reconstruction, get_policy_probs)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(args, epoch, recon_model, model, loader, optimiser, writer, data_range_dict):
    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(loader)

    for it, data in enumerate(loader):
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
        # shape after unsqueeze = batch x channel x columns x rows x complex
        kspace = kspace.unsqueeze(1).to(args.device)
        masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)
        # shape after unsqueeze = batch x channel x columns x rows
        zf = zf.unsqueeze(1).to(args.device)
        gt = gt.unsqueeze(1).to(args.device)
        gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
        gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
        unnorm_gt = gt * gt_std + gt_mean
        data_range = torch.stack([data_range_dict[vol] for vol in fname])

        # Base reconstruction model forward pass
        recon = recon_model(zf)

        optimiser.zero_grad()
        for step in range(args.acquisition_steps):
            # Base score from which to calculate acquisition rewards
            base_score = compute_ssim(recon, unnorm_gt, size_average=False,
                                      data_range=data_range).mean(-1).mean(-1)  # keep channel dim = 1
            # Get policy and probabilities. Sample actions from policy.
            policy, probs = get_policy_probs(model, recon, mask)
            actions = policy.sample((args.num_trajectories,)).transpose(0, 1)

            # Obtain rewards in parallel by taking actions in parallel
            # TODO: Check that this doesn't mutate the mask and masked_kspace
            mask, masked_kspace, zf, recons = compute_next_step_reconstruction(recon_model, kspace,
                                                                               masked_kspace, mask, actions)
            ssim_scores = compute_scores(args, recons, gt_mean, gt_std, unnorm_gt, data_range, comp_psnr=False)
            # batch x num_trajectories
            action_rewards = ssim_scores - base_score
            # batch x num_trajectories
            action_logprobs = torch.log(torch.gather(probs, 1, actions))
            # batch x 1
            avg_reward = torch.mean(action_rewards, dim=1, keepdim=True)
            # batch x k
            if not args.no_baseline:
                # Local baseline
                loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(1) - 1)
            else:
                # No-baseline
                loss = -1 * (action_logprobs * action_rewards) / actions.size(1)
            # batch
            loss = loss.sum(dim=1)

            loss = loss.mean()
            loss.backward()

            epoch_loss[step] += loss.item() / len(loader) * mask.size(0) / args.batch_size
            report_loss[step] += loss.item() / args.report_interval * mask.size(0) / args.batch_size
            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

            if step == len(args.acquisition_steps) - 1:
                continue  # Don't need to actually compute next step reconstruction for the last acquisition

            # Acquire next row by sampling
            next_rows = policy.sample()
            mask, masked_kspace, zf, impro_input = compute_next_step_reconstruction(recon_model, kspace, masked_kspace,
                                                                                    mask, next_rows)

        optimiser.step()

        if it % args.report_interval == 0:
            if it == 0:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, args.report_interval * l * 1e3)
                                      for i, l in enumerate(report_loss)])
            else:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, l * 1e3) for i, l in enumerate(report_loss)])
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}], '
                f'Iter = [{it:4d}/{len(loader):4d}], '
                f'Time = {time.perf_counter() - start_iter:.2f}s, '
                f'Avg Loss per step x1e3 = [{loss_str}] ',
            )

            report_loss = [0. for _ in range(args.acquisition_steps)]

        start_iter = time.perf_counter()

    if args.wandb:
        wandb.log({'train_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}}, step=epoch + 1)

    return np.mean(epoch_loss), time.perf_counter() - start_epoch


def evaluate(args, epoch, recon_model, model, loader, writer, partition, data_range_dict):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """
    model.eval()
    ssims, psnrs = 0, 0
    tbs = 0  # data set size counter
    start = time.perf_counter()
    with torch.no_grad():
        for it, data in enumerate(loader):
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
            # shape after unsqueeze = batch x channel x columns x rows x complex
            kspace = kspace.unsqueeze(1).to(args.device)
            masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
            # shape after unsqueeze = batch x channel x columns x rows
            zf = zf.unsqueeze(1).to(args.device)
            gt = gt.unsqueeze(1).to(args.device)
            gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            unnorm_gt = gt * gt_std + gt_mean
            data_range = torch.stack([data_range_dict[vol] for vol in fname])
            tbs += mask.size(0)

            # Base reconstruction model forward pass
            recon = recon_model(zf)
            unnorm_recon = recon[:, 0:1, :, :] * gt_std + gt_mean
            init_ssim_val = compute_ssim(unnorm_recon, unnorm_gt, size_average=False,
                                         data_range=data_range).mean(dim=(-1, -2)).sum()
            init_psnr_val = compute_psnr(args, unnorm_recon, unnorm_gt, data_range)

            base_mask = mask
            base_masked_kspace = masked_kspace
            base_impro_input = impro_input
            for _ in range(args.num_test_trajectories):
                mask = base_mask.clone()
                masked_kspace = base_masked_kspace.clone()
                impro_input = base_impro_input.clone()

                batch_ssims = [init_ssim_val.item()]
                batch_psnrs = [init_psnr_val.item()]

                for step in range(args.acquisition_steps):
                    policy, probs = get_policy_probs(model, recon, mask)
                    next_rows = policy.sample()
                    mask, masked_kspace, zf, recons = compute_next_step_reconstruction(recon_model, kspace,
                                                                                       masked_kspace, mask, next_rows)
                    ssim_scores, psnr_scores = compute_scores(args, recons, gt_mean, gt_std, unnorm_gt, data_range,
                                                              comp_psnr=True)
                    assert len(ssim_scores.size()) == 2
                    # eventually shape = al_steps
                    batch_ssims.append(ssim_scores.sum().item())
                    batch_psnrs.append(psnr_scores.sum().item())

                # shape of al_steps
                ssims += np.array(batch_ssims)
                psnrs += np.array(batch_psnrs)

    ssims /= (tbs * args.num_test_trajectories)
    psnrs /= (tbs * args.num_test_trajectories)

    # Logging
    if partition in ['Val', 'Train']:
        for step, val in enumerate(ssims):
            writer.add_scalar(f'{partition}SSIM_step{step}', val, epoch)
            writer.add_scalar(f'{partition}PSNR_step{step}', psnrs[step], epoch)

        if args.wandb:
            wandb.log({f'{partition.lower()}_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
            wandb.log({f'{partition.lower()}_psnrs': {str(key): val for key, val in enumerate(psnrs)}}, step=epoch + 1)

    elif partition == 'Test':
        # Only computed once, so loop over all epochs for wandb logging
        if args.wandb:
            for epoch in range(args.num_epochs):
                wandb.log({'val_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
                wandb.log({'val_psnrs': {str(key): val for key, val in enumerate(psnrs)}}, step=epoch + 1)

    else:
        raise ValueError(f"'partition' should be in ['Train', 'Val', 'Test'], not: {partition}")

    return ssims, psnrs, time.perf_counter() - start


# TODO: Separate eval on test data script?
def train_and_eval(args, recon_args, recon_model):
    if args.resume:
        # Check that this works
        resumed = True
        new_run_dir = args.impro_model_checkpoint.parent
        data_path = args.data_path
        # In case models have been moved to a different machine, make sure the path to the recon model is the
        # path provided.
        recon_model_checkpoint = args.recon_model_checkpoint

        model, args, start_epoch, optimiser = load_policy_model(pathlib.Path(args.impro_model_checkpoint), optim=True)

        args.old_run_dir = args.run_dir
        args.old_recon_model_checkpoint = args.recon_model_checkpoint
        args.old_data_path = args.data_path

        args.recon_model_checkpoint = recon_model_checkpoint
        args.run_dir = new_run_dir
        args.data_path = data_path
        args.resume = True
    else:
        resumed = False
        # Improvement model to train
        model = build_policy_model(args)
        # Add mask parameters for training
        args = add_mask_params(args, recon_args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimiser = build_optim(args, model.parameters())
        start_epoch = 0
        # Create directory to store results in
        savestr = 'res{}_al{}_accel{}_{}_{}_k{}_{}_{}'.format(args.resolution, args.acquisition_steps,
                                                              args.accelerations, args.impro_model_name,
                                                              args.recon_model_name, args.num_target_rows,
                                                              datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                              ''.join(choice(ascii_uppercase) for _ in range(5)))
        args.run_dir = args.exp_dir / savestr
        args.run_dir.mkdir(parents=True, exist_ok=False)

    args.resumed = resumed

    if args.wandb:
        allow_val_change = args.resumed  # only allow changes if resumed: otherwise something is wrong.
        wandb.config.update(args, allow_val_change=allow_val_change)
        wandb.watch(model, log='all')

    # Logging
    logging.info(args)
    logging.info(recon_model)
    logging.info(model)
    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    # Parameter counting
    logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(recon_model), count_trainable_parameters(recon_model),
        count_untrainable_parameters(recon_model)))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)
    elif args.scheduler_type == 'multistep':
        if not isinstance(args.lr_multi_step_size, list):
            args.lr_multi_step_size = [args.lr_multi_step_size]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, args.lr_multi_step_size, args.lr_gamma)
    else:
        raise ValueError("{} is not a valid scheduler choice ('step', 'multistep')".format(args.scheduler_type))

    # Create data loaders
    train_loader = create_data_loader(args, 'train', shuffle=True)
    dev_loader = create_data_loader(args, 'val', shuffle=False)

    train_data_range_dict = create_data_range_dict(args, train_loader)
    dev_data_range_dict = create_data_range_dict(args, dev_loader)

    if not args.resume:
        if args.do_train_ssim:
            do_and_log_evaluation(recon_model, model, train_loader, train_data_range_dict, writer, -1, 'Train')
        do_and_log_evaluation(recon_model, model, dev_loader, dev_data_range_dict, writer, -1, 'Val')

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer,
                                             train_data_range_dict)
        logging.info(
            f'Epoch = [{epoch+1:3d}/{args.num_epochs:3d}] TrainLoss = {train_loss:.3g} TrainTime = {train_time:.2f}s '
        )

        if args.do_train_ssim:
            do_and_log_evaluation(recon_model, model, train_loader, train_data_range_dict, writer, epoch, 'Train')
        do_and_log_evaluation(recon_model, model, dev_loader, dev_data_range_dict, writer, epoch, 'Val')

        scheduler.step()
        save_policy_model(args, args.run_dir, epoch, model, optimiser, args.milestones)
    writer.close()


def do_and_log_evaluation(recon_model, model, loader, data_range_dict, writer, epoch, partition):
    ssims, psnrs, score_time = evaluate(args, epoch, recon_model, model, loader, writer, partition, data_range_dict)
    ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(ssims)])
    psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(psnrs)])
    logging.info(f'{partition}SSIM = [{ssims_str}]')
    logging.info(f'{partition}PSNR = [{psnrs_str}]')
    logging.info(f'{partition}ScoreTime = {score_time:.2f}s')


def test(args, recon_model):
    model, policy_args = load_policy_model(pathlib.Path(args.impro_model_checkpoint))

    policy_args.train_state = args.train_state
    policy_args.dev_state = args.dev_state
    policy_args.test_state = args.test_state

    policy_args.wandb = args.wandb
    policy_args.num_test_trajectories = args.num_test_trajectories

    if args.wandb:
        wandb.config.update(args)
        wandb.watch(model, log='all')

        # Initialise summary writer
    writer = SummaryWriter(log_dir=policy_args.run_dir / 'summary')

    # Logging
    logging.info(policy_args)
    logging.info(recon_model)
    logging.info(model)

    # Parameter counting
    logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(recon_model), count_trainable_parameters(recon_model),
        count_untrainable_parameters(recon_model)))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    # Create data loader
    test_loader = create_data_loader(policy_args, 'test', shuffle=False)
    test_data_range_dict = create_data_range_dict(args, test_loader)

    do_and_log_evaluation(recon_model, model, test_loader, test_data_range_dict, writer, -1, 'Test')

    writer.close()


def main(args):
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)

    # Policy model to train
    if args.do_train:
        train_and_eval(args, recon_args, recon_model)
    else:
        test(args, recon_model)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resolution', default=128, type=int, help='Resolution of images')
    parser.add_argument('--dataset', default='knee', help='Dataset type to use.')
    parser.add_argument('--data_path', type=pathlib.Path, required=True,
                        help="Path to the dataset. Make sure to set this consistently with the 'dataset' "
                             "argument above.")

    parser.add_argument('--sample_rate', type=float, default=0.04,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str2none, default=None,
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')
    parser.add_argument('--recon_model_checkpoint', type=pathlib.Path, required=True,
                        help='Path to a pretrained reconstruction model.')
    parser.add_argument('--num_trajectories', type=int, default=8, help='Number of actions to sample every acquisition '
                        'step during training.')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')

    parser.add_argument('--exp_dir', type=pathlib.Path, required=True,
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                        'in provided directory each run')
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--reciprocals_in_center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')
    parser.add_argument('--acquisition_steps', default=16, type=int, help='Acquisition steps to train for per image.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of ConvNet layers. Note that setting '
                        'this too high will cause size mismatch errors, due to even-odd errors in calculation for '
                        'layer size post-flattening (due to max pooling).')
    parser.add_argument('--drop_prob', type=float, default=0, help='Dropout probability')
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Strength of weight decay regularization. TODO: this currently breaks because many weights'
                        'are not updated every step (since we select certain targets only); FIX THIS.')

    parser.add_argument('--center_volume', type=str2bool, default=True,
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--data_parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--do_train_ssim', type=str2bool, default=True,
                        help='Whether to compute SSIM values on training data.')

    # Sweep params
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators. If set to 0, no seed '
                        'will be used.')

    parser.add_argument('--num_chans', type=int, default=16, help='Number of ConvNet channels in first layer.')
    parser.add_argument('--fc_size', default=256, type=int, help='Size (width) of fully connected layer(s).')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--lr_step_size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--scheduler_type', type=str, choices=['step', 'multistep'], default='step',
                        help='Number of training epochs')
    parser.add_argument('--lr_multi_step_size', nargs='+', type=int, default=40,
                        help='Epoch at which to decay the lr if using multistep scheduler.')

    parser.add_argument('--do_train', type=str2bool, default=True,
                        help='Whether to do training or testing.')
    parser.add_argument('--impro_model_checkpoint', type=pathlib.Path, required=True,
                        help='Path to a pretrained impro model.')

    parser.add_argument('--milestones', nargs='+', type=int, default=[0, 9, 19, 29, 39, 49],
                        help='Epochs at which to save model separately.')

    parser.add_argument('--wandb',  type=str2bool, default=False,
                        help='Whether to use wandb logging for this run.')
    parser.add_argument('--num_test_trajectories', type=int, default=1,
                        help='Number of trajectories to use when testing sampling policy.')

    parser.add_argument('--resume',  type=str2bool, default=False,
                        help='Continue training previous run?')
    parser.add_argument('--run_id', type=str2none, default=None,
                        help='Wandb run_id to continue training from.')

    parser.add_argument('--test_multi',  type=str2bool, default=False,
                        help='Test multiple models in one script')
    parser.add_argument('--impro_model_list', nargs='+', type=str, default=[None],
                        help='List of model paths for multi-testing.')

    parser.add_argument('--project',  type=str, default='mrimpro',
                        help='Wandb project name to use.')

    parser.add_argument('--no_baseline', type=str2bool, default=False,
                        help='Whether to not use a reward baseline at all.')

    return parser


def wrap_main(args):
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    args.use_recon_mask_params = False

    args.milestones = args.milestones + [0, args.num_epochs - 1]

    if args.wandb:
        if args.resume:
            assert args.run_id is not None, "run_id must be given if resuming with wandb."
            wandb.init(project=args.project, resume=args.run_id)
            # wandb.restore(run_path=f"mrimpro/{args.run_id}")
        elif args.test_multi:
            wandb.init(project=args.project, reinit=True)
        else:
            wandb.init(project=args.project, config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False

    main(args)


if __name__ == '__main__':
    # To fix known issue with h5py + multiprocessing
    # See: https://discuss.pytorch.org/t/incorrect-data-using-h5py-with-dataloader/7079/2?u=ptrblck
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    base_args = create_arg_parser().parse_args()

    if base_args.test_multi:
        assert not base_args.do_train, "Doing multiple model testing: do_train must be False."
        assert base_args.impro_model_list is not None, "Doing multiple model testing: must have list of impro models."

        for model in base_args.impro_model_list:
            args = copy.deepcopy(base_args)
            args.impro_model_checkpoint = model
            wrap_main(args)
            wandb.join()

    else:
        wrap_main(base_args)
