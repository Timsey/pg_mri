

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
