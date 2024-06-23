import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_channels=1, img_size=28, latent_dim=20):
        super(VAE, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(img_channels * img_size * img_size, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean
        self.fc22 = nn.Linear(400, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, img_channels * img_size * img_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.img_channels * self.img_size * self.img_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
