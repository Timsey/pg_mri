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
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
               f'kernel_size={self.kernel_size}, padding={self.padding})'


class MultiMaskConvModel(nn.Module):
    def __init__(self, resolution, in_chans, chans, depth, sens):
        super().__init__()

        self.resolution = resolution
        self.depth = depth
        self.sens = sens
        self.in_chans = in_chans
        self.chans = chans

        self.kernel_sizes = [1, 3, 5]
        self.padding = [kernel_size // 2 for kernel_size in self.kernel_sizes]

        layers_list = []
        for i, (kernel_size, padding) in enumerate(zip(self.kernel_sizes, self.padding)):
            assert kernel_size % 2 == 1, "Kernel size must be an odd number"
            chans = self.chans
            layers = [Conv1DBlock(in_chans, chans, kernel_size=kernel_size, padding=padding)]
            for _ in range(depth):
                layers.append(Conv1DBlock(chans, chans * 2, kernel_size=kernel_size, padding=padding))
                chans = chans * 2
            for _ in range(depth):
                layers.append(Conv1DBlock(chans, chans // 2, kernel_size=kernel_size, padding=padding))
                chans = chans // 2

            layers_list.append(layers)

        # No ReLU for the last layer
        self.mask_encodings = nn.ModuleList([nn.Sequential(*layers) for layers in layers_list])
        self.out = nn.Sequential(
            Conv1DBlock(self.chans * len(layers_list), self.chans, kernel_size=1, padding=0),
            nn.Conv1d(self.chans, 1, kernel_size=1, padding=0)
        )

    def forward(self, channels, mask):
        # mask = (mask - 0.5) * 2  # -1,1 normalisation
        if self.sens:  # Using sensitivity
            sens_maps = channels[:, 1:3, :, :]
            sens_per_row = torch.mean(sens_maps, dim=(-2))
            if self.in_chans == 2:
                encs = sens_per_row
            elif self.in_chans == 3:  # Also use mask
                encs = torch.cat((mask.unsqueeze(1), sens_per_row), dim=1)
            else:
                raise ValueError("Cannot use sensitivity with 'in_chans' at {}".format(self.in_chans))
        else:  # Not using sensitivity: only mask
            encs = mask.unsqueeze(1)

        enc = [encoding(encs) for encoding in self.mask_encodings]
        enc = self.out(torch.cat(enc, dim=1))
        assert enc.shape[-1] == self.resolution
        return enc.squeeze(1)

    def __repr__(self):
        rep = ''
        for enc in self.mask_encodings:
            rep += str(enc)
        return f'MultiMaskModel({rep}, {self.out})'


def build_impro_multimaskconv_model(args):
    model = MultiMaskConvModel(
        resolution=args.resolution,
        in_chans=args.in_chans,
        chans=args.num_chans,
        depth=args.maskconv_depth,
        sens=args.use_sensitivity
    ).to(args.device)
    return model
