import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, 2),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        )
        self.Tconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)]
        )

    def crop(self, x, features):
        _, _, w, h = x.shape
        return transforms.CenterCrop([w, h])(features)

    def forward(self, x, features):
        for i, block in enumerate(self.decoder_blocks):
            x = self.Tconvs[i](x)
            features_for_cat = self.crop(x, features[i])
            x = torch.cat([x, features_for_cat], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg['encoder_channels'])
        self.decoder = Decoder(cfg['decoder_channels'])
        self.head = nn.Conv2d(cfg['decoder_channels'][-1], cfg['n_classes'], 1)
        self.retain_dim = cfg['retain_dim']
        self.out_size = cfg['out_size']

    def forward(self, x):
        encoded_features = self.encoder(x)
        decoded_features = self.decoder(encoded_features[::-1][0],
                                        encoded_features[::-1][1:])
        res = self.head(decoded_features)
        if self.retain_dim:
            res = F.interpolate(res, self.out_size)
        return res
