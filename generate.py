import torch
import torchvision
import matplotlib.pyplot as plt
import os
from model import VAE

# Create the output directory if it doesn't exist
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Load the trained VAE model
vae = VAE(img_channels=1)
vae.load_state_dict(torch.load('vae_medmnist.pth'))
vae.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
vae.eval()

# Generate new images
with torch.no_grad():
    sample = torch.randn(64, 256).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    generated_images = vae.decode(sample).cpu()

# Save the generated images
for i, img in enumerate(generated_images):
    img = img.squeeze(0)  # Remove batch dimension
    plt.imsave(os.path.join(output_dir, f'image_{i}.png'), img.permute(1, 2, 0).numpy(), cmap='gray')

# Display the generated images
grid_img = torchvision.utils.make_grid(generated_images, nrow=8)
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
