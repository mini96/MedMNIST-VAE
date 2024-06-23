import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os

# Basic VAE model definition
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

# Loss function definition
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Signal handler function
import signal
import sys

def signal_handler(sig, frame):
    print('Training interrupted. Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5
latent_dim = 20

# Ensure the output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Data loaders
transform = transforms.Compose([transforms.ToTensor()])  # Ensure data is in [0, 1] range
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
vae = VAE(img_channels=1, img_size=28, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Save the model checkpoint
torch.save(vae.state_dict(), os.path.join(output_dir, 'vae_mnist.pth'))

# Generate new images and save them
vae.eval()
with torch.no_grad():
    sample = torch.randn(64, latent_dim).to(device)
    generated_images = vae.decode(sample).cpu()
    for i, img in enumerate(generated_images):
        img = img.view(28, 28)  # Reshape the image
        plt.imsave(os.path.join(output_dir, f'image_{i}.png'), img.numpy(), cmap='gray')

# Display the generated images
grid_img = utils.make_grid(generated_images.view(64, 1, 28, 28), nrow=8)
plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
plt.show()
