import torch
from torch import nn


class MaskFCModel(nn.Module):
    def __init__(self, resolution, fc_size):
        super().__init__()

        self.mask_encoding = nn.Sequential(
            nn.Linear(in_features=resolution, out_features=fc_size),
            nn.LeakyReLU(),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(in_features=fc_size, out_features=fc_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_size, out_features=resolution)
        )

    def forward(self, image, mask):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
            mask (torch.Tensor): Input tensor of shape [resolution], containing 0s and 1s

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        # Mask embedding, image not used
        mask_emb = self.mask_encoding(mask)
        return self.fc_out(mask_emb)


def build_impro_maskfc_model(args):
    model = MaskFCModel(
        resolution=args.resolution,
        fc_size=args.fc_size
    ).to(args.device)
    return model
