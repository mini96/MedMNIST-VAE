import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist import INFO, MedMNIST
from torchvision import transforms
from model import VAE, loss_function

# Choose the dataset you want to work with
data_flag = 'chestmnist'
download = True

info = INFO[data_flag]
n_channels = info['n_channels']

DataClass = getattr(MedMNIST, info['python_class'])

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load the dataset
train_dataset = DataClass(split='train', transform=transform, download=download)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the VAE
vae = VAE(img_channels=n_channels)
vae.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training function
def train_vae(model, data_loader, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(data_loader.dataset)}')

train_vae(vae, train_loader, optimizer)

# Save the trained model
torch.save(vae.state_dict(), 'vae_medmnist.pth')
