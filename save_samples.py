import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from medmnist import ChestMNIST

# Set the output directory for sample images
sample_output_dir = 'sample_images'
os.makedirs(sample_output_dir, exist_ok=True)

# Data loader
transform = transforms.Compose([transforms.ToTensor()])
dataset = ChestMNIST(root='./data', split='train', transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Number of samples to save
num_samples = 10

# Save sample images
for i, (data, label) in enumerate(data_loader):
    if i >= num_samples:
        break
    img = data.squeeze(0)  # Remove batch dimension
    plt.imsave(os.path.join(sample_output_dir, f'sample_{i}.png'), img.numpy(), cmap='gray')

print(f'Saved {num_samples} sample images to {sample_output_dir}')
