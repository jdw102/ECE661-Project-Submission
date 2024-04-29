import torch.nn as nn
import numpy as np

class Generator(nn.Module):

    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
                        nn.Linear(latent_size, 128, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
        self.fc2 = nn.Sequential(
                        nn.Linear(128, 256, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
        self.fc3 = nn.Sequential(
                        nn.Linear(256, 512, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
        self.fc4 = nn.Sequential(
                        nn.Linear(512, 1024, bias=True),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
        self.fc5 = nn.Sequential(
                        nn.Linear(1024, 3072, bias=True),
                        nn.Tanh(),
                    )

    def forward(self, z):
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        output = x.view(x.size(0), 3, 32, 32)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(
                            nn.Linear(3072, 1024),
                            nn.LeakyReLU(0.2, inplace=True),
                    )
        self.fc2 = nn.Sequential(
                            nn.Linear(1024,512),
                            nn.LeakyReLU(0.2,inplace=True),
                    )
        self.fc3 = nn.Sequential(
                            nn.Linear(512, 1),
                            nn.Sigmoid(),
                    )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        x = self.fc1(img_flat)
        x = self.fc2(x)
        output = self.fc3(x)
        return output