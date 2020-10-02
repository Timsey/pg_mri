import torch

from src.reconstruction_model.reconstruction_model_def import build_reconstruction_model


def load_recon_model(args):
    if args.recon_model_name == 'zero_filled':  # zero_filled model
        return None, None

    checkpoint = torch.load(args.recon_model_checkpoint)
    recon_args = checkpoint['args']
    recon_model = build_reconstruction_model(recon_args, args)

    # No gradients for this model
    for param in recon_model.parameters():
        param.requires_grad = False

    if recon_args.data_parallel:  # if model was saved with data_parallel
        recon_model = torch.nn.DataParallel(recon_model)
    recon_model.load_state_dict(checkpoint['model'])
    del checkpoint
    return recon_args, recon_model


# def acquire_new_zf(full_kspace, masked_kspace, next_row):
#     # Acquire row
#     cloned_masked_kspace = masked_kspace.clone()
#     # Acquire row for all samples in the batch
#     # shape = (batch_dim, column, row, complex)
#     cloned_masked_kspace[..., next_row, :] = full_kspace[..., next_row, :]
#     zero_filled, mean, std = get_new_zf(cloned_masked_kspace)
#     return zero_filled, mean, std
#
#
# def acquire_new_zf_batch(full_kspace, batch_masked_kspace, batch_next_rows):
#     # shape = batch x 1 x res = col x res = row x 2
#     batch_cloned_masked_kspace = batch_masked_kspace.clone()
#
#     # Acquire row for all samples in the batch
#     # shape = (batch_dim, column, row, complex)
#     for sl, next_row in enumerate(batch_next_rows):
#         batch_cloned_masked_kspace[sl, :, :, next_row, :] = full_kspace[sl, :, :, next_row, :]
#
#     zero_filled, mean, std = get_new_zf(batch_cloned_masked_kspace)
#     return zero_filled, mean, std
#
#
# def acquire_new_zf_exp(k, mk, to_acquire):
#     # Expand masked kspace over channel dimension to prepare for adding all kspace rows to acquire
#     mk_exp = mk.expand(len(to_acquire), -1, -1, -1).clone()  # TODO: .clone() necessary here? Yes?
#     # Acquire row
#     for index, row in enumerate(to_acquire):
#         mk_exp[index, :, row.item(), :] = k[0, :, row.item(), :]
#     # Obtain zero filled image from all len(to_acquire) new kspaces
#     zero_filled_exp, mean_exp, std_exp = get_new_zf(mk_exp)
#     return zero_filled_exp, mean_exp, std_exp
#
#
# def acquire_new_zf_exp_batch(k, mk, to_acquire):
#     # Expand masked kspace over channel dimension to prepare for adding all kspace rows to acquire
#     # TODO: did making the expand --> repeat change really make the difference? Nope, seems to give exact same
#     # mk_exp = mk.expand(-1, to_acquire.size(1), -1, -1, -1).clone()  # TODO: .clone() necessary here? Yes?
#     mk_exp = mk.repeat(1, to_acquire.size(1), 1, 1, 1)
#     # Loop over slices in batch
#     for sl, rows in enumerate(to_acquire):
#         # Loop over indices to acquire
#         for index, row in enumerate(rows):
#             mk_exp[sl, index, :, row.item(), :] = k[sl, 0, :, row.item(), :]
#     # Obtain zero filled image from all new kspaces
#     zero_filled_exp, mean_exp, std_exp = get_new_zf(mk_exp)
#     return zero_filled_exp, mean_exp, std_exp