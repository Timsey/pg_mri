import torch


def reduce(x, reduction="mean"):
    """Batch reduction of a tensor."""
    if reduction == "sum":
        x = x.sum()
    elif reduction == "mean":
        x = x.mean()
    elif reduction == "none":
        x = x
    else:
        raise ValueError("unknown reduction={}.".format(reduction))
    return x


def l1_loss_gradfixed(pred, target, reduction="mean"):
    """Computes the F1 loss with subgradient 0."""
    diff = pred - target
    loss = torch.abs(diff)
    loss = reduce(loss, reduction=reduction)
    return loss


def huber_loss(pred, target, delta=1e-3, reduction="mean"):
    """Computes the Huber loss."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    loss = torch.where(abs_diff < delta,
                       0.5 * diff**2,
                       delta * (abs_diff - 0.5 * delta))
    loss = reduce(loss, reduction=reduction)
    return loss
