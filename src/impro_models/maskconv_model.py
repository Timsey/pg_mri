import torch
from torch import nn


class Conv1DBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, kernel_size, padding):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding

        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
            # nn.InstanceNorm1d(out_chans),  # Does not use batch statistics: unaffected by model.eval()
            nn.ReLU()
        )

    def forward(self, input):
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}'


class MaskConvModel(nn.Module):
    def __init__(self, resolution, in_chans, chans, depth, sens):
        super().__init__()

        self.resolution = resolution
        self.depth = depth
        self.sens = sens
        self.in_chans = in_chans
        self.kernel_size = 5
        assert self.kernel_size % 2 == 1, "Kernel size must be an odd number"
        self.padding = self.kernel_size // 2

        layers = [Conv1DBlock(in_chans, chans, kernel_size=self.kernel_size, padding=self.padding)]
        for _ in range(depth):
            layers.append(Conv1DBlock(chans, chans * 2, kernel_size=self.kernel_size, padding=self.padding))
            chans = chans * 2
        for _ in range(depth):
            layers.append(Conv1DBlock(chans, chans // 2, kernel_size=self.kernel_size, padding=self.padding))
            chans = chans // 2
        # No ReLU for the last layer
        layers.append(nn.Conv1d(chans, 1, kernel_size=self.kernel_size, padding=self.padding))

        self.mask_encoding = nn.Sequential(*layers)

    def forward(self, channels, mask):
        # mask = (mask - 0.5) * 2  # -1,1 normalisation
        if self.sens:  # Using sensitivity
            sens_maps = channels[:, 1:3, :, :]
            sens_per_row = torch.mean(sens_maps, dim=(-2))
            if self.in_chans == 2:
                enc = self.mask_encoding(sens_per_row)
            elif self.in_chans == 3:  # Also use mask
                enc = self.mask_encoding(torch.cat((mask.unsqueeze(1), sens_per_row), dim=1))
            else:
                raise ValueError("Cannot use sensitivity with 'in_chans' at {}".format(self.in_chans))
        else:  # Not using sensitivity: only mask
            enc = self.mask_encoding(mask.unsqueeze(1))
        assert enc.shape[-1] == self.resolution
        return enc.squeeze(1)


def build_impro_maskconv_model(args):
    model = MaskConvModel(
        resolution=args.resolution,
        in_chans=args.in_chans,
        chans=args.num_chans,
        depth=args.maskconv_depth,
        sens=args.use_sensitivity
    ).to(args.device)
    return model
