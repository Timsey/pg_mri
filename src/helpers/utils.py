import json
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def str2none(v):
    if v is None:
        return v
    if v.lower() == 'none':
        return None
    else:
        return v


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0


def count_untrainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad) if model is not None else 0


def add_mask_params(args):
    """Creates list of all combinations of accelerations and center_fractions to be used my MaskFunc, and adds
    these to args.

    E.g.:
    - If accel is 8, then 1/8th of all rows will be sampled on average.
    - If reciprocals_in_center is 2, then half of sampled rows will be in the center.
    """

    # Use supplied mask settings
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


def build_optim(args, params):
    optimiser = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimiser