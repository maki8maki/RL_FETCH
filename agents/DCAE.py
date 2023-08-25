import torch.nn as nn
from agents.utils import size_after_conv, Reshape

class DCAE(nn.Module):
    def __init__(self, image_size, hidden_dim) -> None:
        super().__init__()
        img_height, img_width, img_channel = image_size
        channels = [img_channel, 32, 64, 128, 256]
        after_height = img_height
        after_width = img_width
        ksize = 3
        for _ in range(len(channels)-1):
            after_height = size_after_conv(after_height, ksize=ksize)
            after_width = size_after_conv(after_width, ksize=ksize)
        after_size = after_height * after_width * channels[4]
        features = [after_size, 1000, 150, hidden_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=ksize),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=ksize),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=ksize),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=ksize),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=features[0], out_features=features[1]),
            nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(),
            nn.Linear(in_features=features[1], out_features=features[2]),
            nn.BatchNorm1d(num_features=features[2]),
            nn.ReLU(),
            nn.Linear(in_features=features[2], out_features=features[3]),
            nn.BatchNorm1d(num_features=features[3])
        )
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features[3], out_features=features[2]),
            nn.BatchNorm1d(num_features=features[2]),
            nn.ReLU(),
            nn.Linear(in_features=features[2], out_features=features[1]),
            nn.BatchNorm1d(num_features=features[1]),
            nn.ReLU(),
            nn.Linear(in_features=features[1], out_features=features[0]),
            nn.BatchNorm1d(num_features=features[0]),
            nn.ReLU(),
            Reshape((-1, channels[4], after_height, after_width)),
            nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[3], kernel_size=ksize),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=ksize),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=ksize),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=ksize),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_pred=False):
        h = self.encoder(x)
        if not return_pred:
            return h
        else:
            _h = self.relu(h)
            x_pred = self.decoder(_h)
            return h, x_pred