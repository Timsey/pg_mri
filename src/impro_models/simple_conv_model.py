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


class ImproModel(nn.Module):
    def __init__(self, resolution, in_chans, chans, num_pool_layers, drop_prob):
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
        self.drop_prob = drop_prob

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
        for i in range(num_pool_layers):
            if i == 2:  # First two layers use 4x4 pooling, rest use 2x2 pooling  # TODO: this is badly hardcoded
                self.pool_size = 2
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, pool_size=self.pool_size)]
            self.flattened_size = self.flattened_size * 2 // self.pool_size ** 2
            ch *= 2

        self.fc_recon = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
        )

        self.mask_encoding = nn.Sequential(
            nn.Linear(in_features=resolution, out_features=resolution),
            nn.LeakyReLU(),
            nn.Linear(in_features=resolution, out_features=resolution),
            nn.LeakyReLU(),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(in_features=512 + resolution, out_features=256 + resolution // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=256 + resolution // 2, out_features=resolution)
        )

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
            mask (torch.Tensor): Input tensor of shape [resolution], containing 0s and 1s

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        # Image embedding
        # Initial block
        image_emb = self.channel_layer(image)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            image_emb = layer(image_emb)
        image_emb = self.fc_recon(image_emb.flatten(start_dim=1))  # flatten all but batch dimension
        assert len(image_emb.shape) == 2

        # Mask embedding
        mask_emb = self.mask_encoding(mask)
        assert len(image_emb.shape) == 2

        # First dimension is batch dimension
        # Concatenate among second (last) dimension
        emb = torch.cat((image_emb, mask_emb), dim=-1)
        return self.fc_out(emb)
