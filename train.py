import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os
from medmnist import ChestMNIST
from model import VAE
from vqvae import VQVAE

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
model_type = 'vqvae'  # Choose between 'vae' and 'vqvae'
batch_size = 64
learning_rate = 0.001
num_epochs = 5
latent_dim = 20
num_embeddings = 512
embedding_dim = 64
commitment_cost = 0.25

# Ensure the output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Data loaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = ChestMNIST(root='./data', split='train', transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
if model_type == 'vae':
    model = VAE(img_channels=1, img_size=28, latent_dim=latent_dim).to(device)
elif model_type == 'vqvae':
    model = VQVAE(img_channels=1, img_size=28, num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost).to(device)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if model_type == 'vae':
            recon_batch, mu, logvar = model(data)
            loss = F.binary_cross_entropy(recon_batch, data.view(-1, 28*28), reduction='sum') + \
                   -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        elif model_type == 'vqvae':
            recon_batch, vq_loss, perplexity = model(data)
            recon_loss = F.mse_loss(recon_batch, data)
            loss = recon_loss + vq_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), os.path.join(output_dir, f'{model_type}_chestmnist.pth'))

# Generate new images and save them
model.eval()
with torch.no_grad():
    if model_type == 'vae':
        sample = torch.randn(64, latent_dim).to(device)
        generated_images = model.decode(sample).cpu()
    elif model_type == 'vqvae':
        sample = torch.randn(64, embedding_dim, 7, 7).to(device)
        generated_images = model.decode(sample).cpu()
    for i, img in enumerate(generated_images):
        img = img.view(28, 28)  # Reshape the image
        plt.imsave(os.path.join(output_dir, f'image_{i}.png'), img.numpy(), cmap='gray')

# Display the generated images
grid_img = utils.make_grid(generated_images.view(64, 1, 28, 28), nrow=8)
plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
plt.show()
