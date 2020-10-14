"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import logging
import pathlib
import random
import shutil
import time
import h5py
from collections import defaultdict

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from src.reconstruction_model.reconstruction_model_def import build_reconstruction_model
from src.reconstruction_model.reconstruction_model_utils import (load_recon_model, save_reconstructions, Metrics,
                                                                 METRIC_FUNCS, change_target_resolution)
from src.helpers.utils import build_optim, save_json, str2bool, str2none
from src.helpers.data_loading import create_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    true_avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        _, _, _, input, target, _, _, _, _ = data
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        recon = model(input).squeeze(1)
        loss = F.l1_loss(recon, target)
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        true_avg_loss = (true_avg_loss * iter + loss.mean()) / (iter + 1)
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)
        writer.add_scalar('TrueTrainLossL1', true_avg_loss, global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} AvgLoss = {avg_loss:.4g} TrueAvgLoss = {true_avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate_loss(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    true_avg_loss = 0.
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            _, _, _, input, target, _, _, _, _ = data
            input = input.unsqueeze(1).to(args.device)
            target = target.to(args.device)

            recon = model(input).squeeze(1)
            loss = F.mse_loss(recon, target, reduction='mean')
            l1_loss = (recon - target).abs()
            true_avg_loss = (true_avg_loss * iter + l1_loss.mean()) / (iter + 1)
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
        writer.add_scalar('TrueDevLossL1', true_avg_loss, epoch)
    return np.mean(losses), true_avg_loss, time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.train()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            _, _, _, input, target, _, _, _, _ = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            recon = model(input)
            save_image(target, 'Target')
            save_image(recon, 'Reconstruction')
            save_image(torch.abs(target - recon), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train_unet(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        recon_model, args, start_epoch, optimizer = load_recon_model(args.recon_model_checkpoint, optim=True)
    else:
        model = build_reconstruction_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.exp_dir / 'args.json', args_dict)

    train_loader = create_data_loader(args, 'train', shuffle=True)
    dev_loader = create_data_loader(args, 'val')
    display_loader = create_data_loader(args, 'val', display=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_l1loss, dev_time = evaluate_loss(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)
        scheduler.step()

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainL1Loss = {train_loss:.4g} DevL1Loss = {dev_l1loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def run_unet(args):
    # Evaluate reconstruction model using the settings that it was trained on
    recon_args, model = load_recon_model(args)
    recon_args.data_path = args.data_path  # in case model was trained on different machine
    data_loader = create_data_loader(recon_args, args.partition)

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for _, _, _, input, _, gt_mean, gt_std, fnames, slices in data_loader:
            input = input.unsqueeze(1).to(args.device)
            recons = model(input).squeeze(1).to('cpu')
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * gt_std[i] + gt_mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    args.predictions_path = args.recon_model_checkpoint.parent / 'reconstructions'
    save_reconstructions(reconstructions, args.predictions_path)


def evaluate(args):
    # Use esc for Knee data, rss for Brain data (since it's technically multicoil)
    recons_key = 'reconstruction_esc' if args.dataset == 'knee' else 'reconstruction_rss'
    metrics = Metrics(METRIC_FUNCS)
    recons_files = [path.name for path in args.predictions_path.iterdir()]

    # This path is partially hardcoded right now
    args.target_path = args.data_path / f'singlecoil_{args.partition}'
    for tgt_file in args.target_path.iterdir():
        if tgt_file.name not in recons_files:
            continue
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:
            if args.acquisition is not None and args.acquisition != target.attrs['acquisition']:
                continue
            target = target[recons_key].value
            target = change_target_resolution(args, target)
            if args.center_volume:
                num_slices = target.shape[0]
                target = target[num_slices // 4: 3 * num_slices // 4, :, :]
            recons = recons['reconstruction'].value
            metrics.push(target, recons)

    with open(args.predictions_path / "metrics.txt", "w") as text_file:
        print(f"{metrics}", file=text_file)
    print(metrics)


def main(args):
    logging.info(args)
    if args.do_train:
        train_unet(args)
    else:
        run_unet(args)
        evaluate(args)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=128, type=int, help='Resolution of images')
    parser.add_argument('--dataset', default='knee', type=str, choices=['knee', 'brain'],
                        help='Dataset to use.')
    parser.add_argument('--data_path', type=pathlib.Path, required=True,
                        help='Path to the dataset')
    parser.add_argument('--sample_rate', type=float, default=1.,
                        help='Fraction of total volumes to include')
    parser.add_argument('--accelerations', nargs='+', default=[4, 4, 4, 6, 6, 8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                        'provided, then one of those is chosen uniformly at random for each volume.')
    parser.add_argument('--center_fractions', nargs='+', default=[0.25, 0.167, 0.125, 0.167, 0.125, 0.125], type=float,
                        help='Fraction of low-frequency k-space columns to be sampled. Should '
                        'have the same length as accelerations')
    parser.add_argument('--num_pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop_prob', type=float, default=0, help='Dropout probability')  # 0.2 in Kendall&Gal
    parser.add_argument('--num_chans', type=int, default=16, help='Number of U-Net channels')
    parser.add_argument('--val_batch_size', default=64, type=int, help='Mini batch size for validation')
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')  # 1e-3 in Kendall&Gal, fastMRI base
    parser.add_argument('--lr_step_size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0,  # 1e-4 in Kendall&Gal (replaces dropout regularis)
                        help='Strength of weight decay regularization')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data_parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default=None,
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--recon_model_checkpoint" should be set with this')
    parser.add_argument('--recon_model_checkpoint', type=pathlib.Path,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--center_volume', type=str2bool, default=True,
                        help='If set, only the center slices of a volume will be included in the dataset.')
    parser.add_argument('--acquisition', type=str2none, default=None,
                        help='If set, only volumes of the specified acquisition type are used '
                             'for evaluation. By default, all volumes are included.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for data loading')

    parser.add_argument('--do_train', type=str2bool, default=True,
                        help='Whether to train or evaluate / test.')
    parser.add_argument('--partition', type=str, default='val', choices=['val', 'test'],
                        help='Partition to evaluate model on (used with do_train=False).')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
