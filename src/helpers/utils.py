import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_untrainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def add_mask_params(args, recon_args):
    """Creates list of all combinations of accelerations and center_fractions to be used my MaskFunc, and adds
    these to args.

    E.g.:
    - If accel is 8, then 1/8th of all rows will be sampled on average.
    - If reciprocals_in_center is 2, then half of sampled rows will be in the center.
    """

    # Use mask settings from reconstruction model
    if args.use_recon_mask_params:
        args.accelerations = recon_args.accelerations
        args.center_fractions = recon_args.center_fractions
        args.reciprocals_in_center = recon_args.mask_in_center_fracs

    # Use supplied mask settings
    else:
        all_accels = []
        all_center_fracs = []
        all_reciprocals_in_center = []
        for accel in args.accelerations:
            for frac in args.reciprocals_in_center:
                all_accels.append(accel)
                all_reciprocals_in_center.append(frac)
                all_center_fracs.append(1 / (frac * accel))

        args.accelerations = all_accels
        args.center_fractions = all_center_fracs
        args.reciprocals_in_center = all_reciprocals_in_center

    return args


def check_args_consistency(args, recon_args):
    assert args.resolution == recon_args.resolution, ("Resolution mismatch between reconstruction model and provided: "
                                                      "{} and {}".format(args.resolution, recon_args.resolution))
    assert args.challenge == recon_args.challenge, ("Challenge mismatch between reconstruction model and provided: "
                                                    "{} and {}".format(args.challenge, recon_args.challenge))


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8
    """

    figure = plt.figure(figsize=(10, 8))
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    return figure
