from torch import nn


class Conv1DBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=5, padding=2),
            # nn.InstanceNorm1d(out_chans),  # Does not use batch statistics: unaffected by model.eval()
            nn.ReLU()
        )

    def forward(self, input):
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}'


class MaskConvModel(nn.Module):
    def __init__(self, resolution, chans, depth):
        super().__init__()

        self.resolution = resolution
        self.depth = depth

        layers = [Conv1DBlock(1, chans)]
        for _ in range(depth):
            layers.append(Conv1DBlock(chans, chans * 2))
            chans = chans * 2
        for _ in range(depth):
            layers.append(Conv1DBlock(chans, chans // 2))
            chans = chans // 2
        # No ReLU for the last layer
        layers.append(nn.Conv1d(chans, 1, kernel_size=5, padding=2))

        self.mask_encoding = nn.Sequential(*layers)

    def forward(self, image, mask):
        # mask = (mask - 0.5) * 2  # -1,1 normalisation
        enc = self.mask_encoding(mask.unsqueeze(1)).squeeze(1)
        assert enc.shape[-1] == self.resolution
        return enc


def build_impro_maskconv_model(args):
    model = MaskConvModel(
        resolution=args.resolution,
        chans=args.num_chans,
        depth=args.maskconv_depth
    ).to(args.device)
    return model
