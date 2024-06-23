import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medmnist import ChestMNIST
from model import CVAE
import signal
import sys

# Signal handler function
def signal_handler(sig, frame):
    print('Training interrupted. Exiting gracefully...')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 50
latent_dim = 256
label_dim = 2  # Adjust this based on the number of classes in ChestMNIST

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ChestMNIST(root='data', split='train', download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ChestMNIST(root='data', split='test', download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
cvae = CVAE(img_channels=1, label_dim=label_dim, latent_dim=latent_dim).to(device)

# Loss and optimizer
optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)
criterion = nn.BCELoss(reduction='sum')

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training function
def train(epoch):
    cvae.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=label_dim).float().to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = cvae(data, target_one_hot)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Test function
def test(epoch):
    cvae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=label_dim).float().to(device)

            recon_batch, mu, logvar = cvae(data, target_one_hot)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
                torchvision.utils.save_image(comparison.cpu(), f'results/reconstruction_{epoch}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

# Main training loop
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, latent_dim).to(device)
        labels = torch.eye(label_dim).repeat(64 // label_dim + 1, 1)[:64].to(device)
        sample = cvae.decode(sample, labels).cpu()
        torchvision.utils.save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png')
    
    # Save model checkpoint
    torch.save(cvae.state_dict(), f'models/cvae_medmnist_epoch_{epoch}.pth')
