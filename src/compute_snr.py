import pathlib
import pickle
import os
import copy
import json
import argparse
import datetime
import random
import numpy as np
from pprint import pprint
from collections import defaultdict

import torch

from src.helpers.data_loading import create_data_loader
from src.helpers.utils import load_json, save_json, str2bool
from src.reconstruction_model.reconstruction_model_utils import load_recon_model
from src.policy_model.policy_model_utils import build_optim, create_data_range_dict, compute_backprop_trajectory
from src.policy_model.policy_model_def import build_policy_model


def load_policy_model(checkpoint_file):
    """
    Loads pretrained policy model, and disables gradients for all but the last layer weights.

    :param checkpoint_file: path to policy model checkpoint.
    :return: (policy model, policy model Arguments object,
              optimizer: PyTorch optimiser stored with policy model)
    """
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_policy_model(args)

    # Only store gradients for final layer
    num = len(list(model.named_parameters()))
    for i, (name, param) in enumerate(model.named_parameters()):
        if i in [num - 1, num - 2]:  # weight and biases of last layer
            assert name in ['fc_out.4.weight', 'fc_out.4.bias'], ("Parameter for which to register gradients has "
                                                                  "unexpected name. This can happen due to a change "
                                                                  "in the policy model definition. If this is intended "
                                                                  "this assert can be removed, but make sure that the "
                                                                  "correct parameters are given a gradient. This "
                                                                  "should then also be fixed at the bottom of "
                                                                  "compute_gradients().")
            param.requires_grad = True
        else:
            param.requires_grad = False

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint

    return model, args, optimizer


def snr_from_grads(args, grads):
    """
    Computes average SNR as well as SNR standard deviation using gradients stored over a number of runs.

    :param args: Arguments object, containing SNR computation hyperparameters.
    :param grads: numpy array containing gradients per batch, of shape
                  (num_batches * data_runs, last_layer_size, second_to_last_layer_size + 1).
                  Assumes the row order in contiguous w.r.t the various data_runs (i.e. epochs). This function
                  automatically detects the number of batches (gradients) stored, and separates them out in groups
                  corresponding to a particular run over the data (epoch).  SNR is computed separately for each run,
                  and then averaged.
    :return: (float: SNR mean, float: SNR std)
    """
    snr_list = []
    assert grads.shape[0] % args.data_runs == 0, ('Something went wrong with concatenating gradients over runs. This '
                                                  'is usually the result of a previous run having been aborted halfway '
                                                  'through finishing a computation. Check that gradients stored for '
                                                  'the last epoch mentioned in the log (above) are complete. If not: ' 
                                                  'delete the relevant epoch{}_t{}_runs{}_batch{}_bs{} directory and '
                                                  'rerun this script.')
    grads_per_run = grads.shape[0] // args.data_runs
    for i in range(grads.shape[0] // grads_per_run):
        # Grab gradients belonging to a single run
        g = grads[grads_per_run * i:grads_per_run * (i + 1), :]
        mean = np.mean(g, axis=0, keepdims=True)
        var = np.mean((g - mean) ** 2, axis=0, keepdims=True)

        # variance of mean = variance of sample / num_samples (batches)
        var = var / g.shape[0]
        snr = np.linalg.norm(mean) / np.linalg.norm(np.sqrt(var))
        snr_list.append(snr)

    snr = np.mean(snr_list)
    snr_std = np.std(snr_list, ddof=1)

    return snr, snr_std


def compute_snr(args, weight_path, bias_path):
    """
    Wrapper for SNR computation. Loads stored gradients and creates gradient array for further computation.

    :param weight_path: str or pathlib.Path, path to stored last layer weight gradients.
    :param bias_path: str or pathlib.Path, path to stored last layer bias node gradients.
    :return: (float: SNR mean, float: SNR std)
    """
    with open(weight_path, 'rb') as f:
        weight_list = pickle.load(f)
    with open(bias_path, 'rb') as f:
        bias_list = pickle.load(f)
    # num_batches x last_layer_size x second_to_last_layer_size
    weight_grads = np.stack(weight_list)
    # num_batches x last_layer_size x 1 (after reshape)
    bias_grads = np.stack(bias_list)[:, :, None]
    # num_batches x last_layer_size x (second_to_last_layer_size + 1)
    grads = np.concatenate((weight_grads, bias_grads), axis=-1)
    snr, std = snr_from_grads(args, grads)
    return snr, std


def add_base_args(args, policy_args):
    """
    Utility to add / overwrite relevant SNR computation hyperparameters to policy model arguments.

    """
    # Batch and trajectory parameters have to be set to match those in args.
    policy_args.batch_size = args.batch_size
    policy_args.batches_step = args.batches_step
    policy_args.num_trajectories = args.num_trajectories

    # Fix paths to those on the running machine (in case model was trained on different machine)
    policy_args.policy_model_checkpoint = args.policy_model_checkpoint
    policy_args.recon_model_checkpoint = args.recon_model_checkpoint
    policy_args.data_path = args.data_path


def compute_gradients(args, epoch):
    """
    Big function that runs a single epoch of training to compute gradients for the SNR calculation. Stores gradients
    in policy model directory, and contains checks for already stored gradients to speed up computations.

    :param args: Arguments object, containing hyperparameters for training epoch.
    :param epoch: current training epoch (i.e. milestone).
    :return: (str: path to stored weight gradients, str: path to stored bias gradients,
              str: parent directory for these two files)
    """
    param_dir = (f'epoch{epoch}_t{args.num_trajectories}'
                 f'_runs{args.data_runs}_batch{args.batch_size}_bs{args.batches_step}')
    param_dir = args.policy_model_checkpoint.parent / param_dir
    param_dir.mkdir(parents=True, exist_ok=True)

    # Create storage path
    weight_path = param_dir / f'weight_grads_r{args.data_runs}.pkl'
    bias_path = param_dir / f'bias_grads_r{args.data_runs}.pkl'
    # Check if already computed (skip computing again if not args.force_computation)
    if weight_path.exists() and bias_path.exists() and not args.force_computation:
        print(f'Gradients already stored in: \n    {weight_path}\n    {bias_path}')
        return weight_path, bias_path, param_dir
    else:
        print('Exact job gradients not already stored. Checking same params but higher number of runs...')

    # Check if all gradients already stored in file for more runs
    for r in range(1, 11, 1):  # Check up to 10 runs
        tmp_param_dir = (f'epoch{epoch}_t{args.num_trajectories}'
                         f'_runs{r}_batch{args.batch_size}_bs{args.batches_step}')
        tmp_weight_path = args.policy_model_checkpoint.parent / tmp_param_dir / f'weight_grads_r{r}.pkl'
        tmp_bias_path = args.policy_model_checkpoint.parent / tmp_param_dir / f'bias_grads_r{r}.pkl'
        # If computation already stored for a higher number of runs, just grab the relevant bit and do not recompute.
        if tmp_weight_path.exists() and tmp_bias_path.exists() and not args.force_computation:
            print(f'Gradients up to run {r} already stored in: \n    {tmp_weight_path}\n    {tmp_bias_path}')
            with open(tmp_weight_path, 'rb') as f:
                full_weight_grads = pickle.load(f)
            with open(tmp_bias_path, 'rb') as f:
                full_bias_grads = pickle.load(f)

            # Get relevant bit for the number of runs requested
            assert len(full_weight_grads) % r == 0, 'Something went wrong with stored gradient shape.'
            grads_per_run = len(full_weight_grads) // r
            weight_grads = full_weight_grads[:grads_per_run * args.data_runs]
            bias_grads = full_bias_grads[:grads_per_run * args.data_runs]

            print(f" Saving only grads of run {args.data_runs} to: \n       {param_dir}")
            with open(weight_path, 'wb') as f:
                pickle.dump(weight_grads, f)
            with open(bias_path, 'wb') as f:
                pickle.dump(bias_grads, f)
            return weight_path, bias_path, param_dir

    # Initialise
    start_run = 0
    weight_grads = []
    bias_grads = []

    # Check if some part of the gradients already computed
    for r in range(args.data_runs, 0, -1):
        tmp_param_dir = (f'epoch{epoch}_t{args.num_trajectories}'
                         f'_runs{r}_batch{args.batch_size}_bs{args.batches_step}')
        tmp_weight_path = args.policy_model_checkpoint.parent / tmp_param_dir / f'weight_grads_r{r}.pkl'
        tmp_bias_path = args.policy_model_checkpoint.parent / tmp_param_dir / f'bias_grads_r{r}.pkl'
        # If part already computed, skip this part of the computation by setting start_run to the highest
        # computed run. Also load the weights.
        if tmp_weight_path.exists() and tmp_bias_path.exists() and not args.force_computation:
            print(f'Gradients up to run {r} already stored in: \n    {tmp_weight_path}\n    {tmp_bias_path}')
            with open(tmp_weight_path, 'rb') as f:
                weight_grads = pickle.load(f)
            with open(tmp_bias_path, 'rb') as f:
                bias_grads = pickle.load(f)
            start_run = r
            break

    model, policy_args, optimiser = load_policy_model(args.policy_model_checkpoint)
    add_base_args(args, policy_args)
    recon_args, recon_model = load_recon_model(policy_args)

    loader = create_data_loader(policy_args, 'train', shuffle=True)
    data_range_dict = create_data_range_dict(policy_args, loader)

    for r in range(start_run, args.data_runs):
        print(f"\n    Run {r + 1} ...")
        cbatch = 0
        for it, data in enumerate(loader):  # Randomly shuffled every time
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, sl_idx = data
            cbatch += 1
            # shape after unsqueeze = batch x channel x columns x rows x complex
            kspace = kspace.unsqueeze(1).to(policy_args.device)
            masked_kspace = masked_kspace.unsqueeze(1).to(policy_args.device)
            mask = mask.unsqueeze(1).to(policy_args.device)
            # shape after unsqueeze = batch x channel x columns x rows
            zf = zf.unsqueeze(1).to(policy_args.device)
            gt = gt.unsqueeze(1).to(policy_args.device)
            gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(policy_args.device)
            gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(policy_args.device)
            unnorm_gt = gt * gt_std + gt_mean
            data_range = torch.stack([data_range_dict[vol] for vol in fname])
            recons = recon_model(zf)

            if cbatch == 1:
                optimiser.zero_grad()

            action_list = []
            logprob_list = []
            reward_list = []
            for step in range(policy_args.acquisition_steps):  # Loop over acquisition steps
                loss, mask, masked_kspace, recons = compute_backprop_trajectory(policy_args, kspace, masked_kspace,
                                                                                mask, unnorm_gt, recons, gt_mean,
                                                                                gt_std, data_range, model, recon_model,
                                                                                step, action_list, logprob_list,
                                                                                reward_list)

            if cbatch == policy_args.batches_step:
                # Store gradients for SNR
                num = len(list(model.named_parameters()))
                for i, (name, param) in enumerate(model.named_parameters()):
                    if i == num - 1:  # biases of last layer
                        weight_grads.append(param.grad.cpu().numpy())
                    elif i == num - 2:  # weights of last layer
                        bias_grads.append(param.grad.cpu().numpy())
                cbatch = 0

        print(f"    - Adding grads of run {r + 1} to: \n       {param_dir}")
        with open(weight_path, 'wb') as f:
            pickle.dump(weight_grads, f)
        with open(bias_path, 'wb') as f:
            pickle.dump(bias_grads, f)

    return weight_path, bias_path, param_dir


def main(base_args):
    """
    Wrapper for gradient computation and SNR calculation. Saves SNR results to separate datetime-stamped directory in
    the current working directory.

    :param base_args: Arguments object containing gradient and SNR computation hyperparameters.
    """
    results_dict = defaultdict(lambda: defaultdict(dict))

    runs = base_args.data_runs
    traj = base_args.num_trajectories

    for i, run_dir in enumerate(base_args.policy_model_dir_list):
        args_dict = load_json(base_args.base_policy_model_dir / run_dir / 'args.json')
        if args_dict['model_type'] == 'greedy':
            mode = 'greedy'
            label = 'greedy'
        else:
            mode = 'nongreedy'
            label = args_dict.get('gamma', None)

        sr = json.loads(args_dict['sample_rate'])
        accels = json.loads(args_dict['accelerations'])
        steps = json.loads(args_dict['acquisition_steps'])
        assert len(accels) == 1, "Using models trained with various accelerations is not supported!"
        accel = accels[0]

        for j, epoch in enumerate(base_args.epochs):
            args = copy.deepcopy(base_args)
            args.mode = mode

            if epoch != max(base_args.epochs):
                args.policy_model_checkpoint = base_args.base_policy_model_dir / run_dir / 'model_{}.pt'.format(epoch)
            else:  # Last epoch model is not always stored separately depending on logging details
                args.policy_model_checkpoint = base_args.base_policy_model_dir / run_dir / 'model.pt'

            pr_str = (f"Job {i*len(base_args.epochs)+j+1}/{len(base_args.policy_model_dir_list) * len(base_args.epochs)}"
                      f"\n   mode: {mode:>9}, accel: {accel:>2}, steps: {steps:>2}, label: {label},\n"
                      f"   ckpt: {epoch:>2}, runs: {runs:>2}, srate: {sr:>3}, traj: {traj:>2}")
            print(pr_str)

            weight_path, bias_path, param_dir = compute_gradients(args, epoch)
            snr, std = compute_snr(args, weight_path, bias_path)

            summary_dict = {'snr': str(snr),
                            'snr_std': str(std),
                            'weight_grads': str(weight_path),
                            'bias_grads': str(bias_path)}

            summary_path = param_dir / f'snr_summary.json'
            print(f"   Saving summary to {summary_path}")
            save_json(summary_path, summary_dict)

            results_dict[run_dir][f'Epoch: {epoch}'] = {'job': (mode, traj, runs, sr, accel, steps, label),
                                                        'snr': str(snr),
                                                        'snr_std': str(std)}
            print(f'SNR: {snr}, STD: {std}')

    savestr = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.json'
    save_dir = pathlib.Path(os.getcwd()) / f'snr_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / savestr

    print('\nFinal results:')
    pprint(results_dict)

    print(f'\nSaving results to: {save_file}')
    save_json(save_file, results_dict)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=pathlib.Path, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--recon_model_checkpoint', type=pathlib.Path, required=True,
                        help='Path to a pretrained reconstruction model. If None then recon-model-name should be'
                        'set to zero_filled.')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators '
                                                            'Set to 0 to use random seed.')

    parser.add_argument('--data_runs', type=int, default=3,
                        help='Number of times to run same SNR experiment for averaging.')
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--batches_step', type=int, default=1,
                        help='Number of batches to compute before doing an optimizer step.')
    parser.add_argument('--num_trajectories', type=int, default=16,
                        help='Number of trajectories to sample SNR.')

    parser.add_argument('--epochs', nargs='+', type=int, default=[0, 9, 19, 29, 39, 49],
                        help='Epochs at which to calculate SNR.')
    parser.add_argument('--force_computation', type=str2bool, default=False,
                        help='Whether to force recomputing SNR if already stored on disk.')

    parser.add_argument('--policy_model_dir_list', nargs='+', type=str, default=[None],
                        help='List of policy model dirs for models to calculate SNR for.')
    parser.add_argument('--base_policy_model_dir', type=pathlib.Path, default=None,
                        help='Base dir for policy models.')

    return parser


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    base_args = create_arg_parser().parse_args()

    if base_args.seed != 0:
        random.seed(base_args.seed)
        np.random.seed(base_args.seed)
        torch.manual_seed(base_args.seed)
        if base_args.device == 'cuda':
            torch.cuda.manual_seed(base_args.seed)

    main(base_args)