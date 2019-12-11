import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModelParam(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        # Encoder
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)
        vch = ch

        # Mean decoder
        self.up_sample_mean = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_mean += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_mean += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        # Log variance decoder
        self.up_sample_logvar = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_logvar += [ConvBlock(vch * 2, vch // 2, drop_prob)]
            vch //= 2
        self.up_sample_logvar += [ConvBlock(vch * 2, vch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(vch, vch // 2, kernel_size=1),
            nn.Conv2d(vch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            Mean (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
            Log variance (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        meanstack, varstack = [], []
        enc = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            enc = layer(enc)
            meanstack.append(enc)
            varstack.append(enc)
            enc = F.max_pool2d(enc, kernel_size=2)

        enc = self.conv(enc)
        meanenc = enc
        logvarenc = enc.clone()

        # Apply up-sampling layers for mean
        for layer in self.up_sample_mean:
            meanenc = F.interpolate(meanenc, scale_factor=2, mode='bilinear', align_corners=False)
            meanenc = torch.cat([meanenc, meanstack.pop()], dim=1)
            meanenc = layer(meanenc)

        # Apply up-sampling layers for mean
        for layer in self.up_sample_logvar:
            logvarenc = F.interpolate(logvarenc, scale_factor=2, mode='bilinear', align_corners=False)
            logvarenc = torch.cat([logvarenc, varstack.pop()], dim=1)
            logvarenc = layer(logvarenc)

        return self.conv2(meanenc), self.conv2(logvarenc)


class Arguments:
    """
    Required to load the reconstruction model. Pickle requires the class definition to be visible/importable
    when loading a checkpoint containing an instance of that class.
    """
    def __init__(self):
        pass


def build_dist_model(recon_args, args):
    gauss_model = UnetModelParam(
        in_chans=1,
        out_chans=1,
        chans=recon_args.num_chans,
        num_pool_layers=recon_args.num_pools,
        drop_prob=recon_args.drop_prob
    ).to(args.device)
    return gauss_model

