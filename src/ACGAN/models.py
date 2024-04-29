import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.input_dim = latent_dim + num_classes
        self.fc = nn.Linear(latent_dim + num_classes, 384)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 96, 6, 2, 0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, 6, 2, 0, bias=False),
            nn.Tanh(),
        )


    def forward(self, input):
        input = input.view(-1, self.input_dim)
        x = self.fc(input).view(-1, 384, 1, 1)
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )
        self.disc = nn.Sequential(
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )
        self.aux = nn.Sequential(
            nn.Linear(8192, num_classes),
            nn.Softmax()
        )


    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 8192)
        return self.disc(x).view(-1, 1), self.aux(x)
