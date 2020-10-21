import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from piq import psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, data_range=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def compute_ssim(img1, img2, window_size=11, size_average=True, data_range=None):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, data_range)


def compute_psnr(args, unnorm_recons, gt_exp, data_range):
    # Have to reshape to batch . trajectories x res x res and then reshape back to batch x trajectories x res x res
    # because of psnr implementation
    psnr_recons = torch.clamp(unnorm_recons, 0., 10.).reshape(gt_exp.size(0) * gt_exp.size(1), 1, args.resolution,
                                                              args.resolution).to('cpu')
    psnr_gt = gt_exp.reshape(gt_exp.size(0) * gt_exp.size(1), 1, args.resolution, args.resolution).to('cpu')
    # First duplicate data range over trajectories, then reshape: this to ensure alignment with recon and gt.
    psnr_data_range = data_range.expand(-1, gt_exp.size(1), -1, -1)
    psnr_data_range = psnr_data_range.reshape(gt_exp.size(0) * gt_exp.size(1), 1, 1, 1).to('cpu')
    psnr_scores = psnr(psnr_recons, psnr_gt, reduction='none', data_range=psnr_data_range)
    psnr_scores = psnr_scores.reshape(gt_exp.size(0), gt_exp.size(1))
    return psnr_scores