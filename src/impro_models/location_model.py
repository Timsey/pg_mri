import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, pool_size=2):
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
        self.pool_size = pool_size

        layers = [nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                  nn.InstanceNorm2d(out_chans),  # Does not use batch statistics: unaffected by model.eval()
                  nn.ReLU(),
                  nn.Dropout2d(drop_prob)]  # To do MC dropout, need model.train() at eval time.

        if pool_size:
            layers.append(nn.MaxPool2d(pool_size))

        self.layers = nn.Sequential(*layers)

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
            f'drop_prob={self.drop_prob}, max_pool_size={self.pool_size})'


class Conv1DBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, kernel_size=5, padding=2):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.padding = padding

        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layers(input)

    def __repr__(self):
        return f'Conv1DBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class Conv1DEncoder(nn.Module):
    def __init__(self, resolution, chans, num_up_layers):
        super().__init__()

        self.resolution = resolution
        self.num_up_layers = num_up_layers

        layers = [Conv1DBlock(1, chans)]
        for _ in range(num_up_layers):
            layers.append(Conv1DBlock(chans, chans * 2))
            chans = chans * 2

        self.mask_encoding = nn.Sequential(*layers)
        self.chans = chans

    def forward(self, mask):
        # mask = (mask - 0.5) * 2  # -1,1 normalisation
        enc = self.mask_encoding(mask.unsqueeze(1)).squeeze(1)
        assert enc.shape[-1] == self.resolution
        return enc


class ConvOutDecoder(nn.Module):
    def __init__(self, resolution, in_chans, num_down_layers):
        super().__init__()

        self.resolution = resolution
        self.in_chans = in_chans

        layers = [Conv1DBlock(in_chans, in_chans // 2)]
        chans = in_chans // 2
        for _ in range(num_down_layers):
            layers.append(Conv1DBlock(chans, chans // 2))
            chans = chans // 2
        layers.append(Conv1DBlock(chans, 1))
        layers.append(nn.Conv1d(1, 1, kernel_size=5, padding=2))

        self.decoding = nn.Sequential(*layers)

    def forward(self, mask):
        dec = self.decoding(mask)
        assert len(dec.shape) == 3  # batch x channel x resolution
        assert dec.shape[-1] == self.resolution
        assert dec.shape[1] == 1  # 1 channel
        return dec


class ImageConvEncoder(nn.Module):
    def __init__(self, resolution, in_chans, chans, num_pool_layers, four_pools, drop_prob, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            resolution (int): Number of neurons in the output FC layer (equal to image number of rows in kspace).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.resolution = resolution
        self.in_chans = in_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.four_pools = four_pools
        self.drop_prob = drop_prob
        self.out_chans = out_chans

        # Size of image encoding after flattening of convolutional output
        # There are 1 + num_pool_layers blocks
        # The first block increases channels from in_chans to chans, without any pooling
        # Every block except the first 2x2 max pools, reducing output by a factor 4
        # Every block except the first doubles channels, increasing output by a factor 2

        self.pool_size = 4
        self.flattened_size = resolution ** 2 * chans

        # Initial from in_chans to chans
        self.channel_layer = ConvBlock(in_chans, chans, drop_prob, pool_size=False)
        # Downsampling convolution
        # These are num_pool_layers layers where each layers 2x2 max pools, and doubles the number of channels
        self.down_sample_layers = nn.ModuleList([])
        ch = chans
        for i in range(num_pool_layers - 1):
            if i == self.four_pools:  # First two layers use 4x4 pooling, rest use 2x2 pooling
                self.pool_size = 2
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, pool_size=self.pool_size)]
            ch *= 2

        layers = []
        # from 100 pixels down to 80
        for _ in range(2):
            layers.append(Conv1DBlock(ch, ch, kernel_size=7, padding=0))
        for _ in range(2):
            layers.append(Conv1DBlock(ch, ch, kernel_size=5, padding=0))
        layers.append(Conv1DBlock(ch, out_chans))
        self.conv1d_layers = nn.Sequential(*layers)

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        # Image embedding
        # Initial block
        image_emb = self.channel_layer(image)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            image_emb = layer(image_emb)
        image_emb = image_emb.flatten(start_dim=2)  # flatten pixel dimensions
        image_emb = self.conv1d_layers(image_emb)

        assert len(image_emb.shape) == 3
        return image_emb


class LocationModel(nn.Module):
    def __init__(self, resolution, in_chans, chans, num_pool_layers, four_pools, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            resolution (int): Number of neurons in the output FC layer (equal to image number of rows in kspace).
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.num_up_sample_layers = 3

        self.mask_encoding = Conv1DEncoder(resolution, chans, self.num_up_sample_layers)
        self.recon_encoding = ImageConvEncoder(resolution, in_chans, chans, num_pool_layers, four_pools,
                                               drop_prob, self.mask_encoding.chans)

        up_chans = chans * 2 ** (self.num_up_sample_layers + 1)
        self.conv_out = ConvOutDecoder(resolution, up_chans, self.num_up_sample_layers)

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
            mask (torch.Tensor): Input tensor of shape [resolution], containing 0s and 1s

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        image = self.recon_encoding(image)
        mask = self.mask_encoding(mask)

        comb = torch.cat((image, mask), dim=1)  # concatenate along channels
        return self.conv_out(comb).squeeze(1)  # remove channel dimension after combining to 1


def build_impro_location_model(args):
    model = LocationModel(
        resolution=args.resolution,
        in_chans=args.in_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        four_pools=args.of_which_four_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model
