import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, img_channels=1, label_dim=10, latent_dim=256):
        super(CVAE, self).__init__()
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(img_channels + label_dim, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(128 * 7 * 7, latent_dim)
        self.enc_fc2 = nn.Linear(128 * 7 * 7, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim + label_dim, 128 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x, y):
        # Concatenate image and label
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.enc_fc1(x)
        logvar = self.enc_fc2(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # Concatenate latent vector and label
        z = torch.cat([z, y], dim=1)
        z = F.relu(self.dec_fc(z))
        z = z.view(z.size(0), 128, 7, 7)
        z = F.relu(self.dec_conv1(z))
        z = torch.sigmoid(self.dec_conv2(z))
        return z

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
