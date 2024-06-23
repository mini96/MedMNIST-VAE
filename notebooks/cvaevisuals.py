{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Conditional Variational Autoencoder (CVAE) Visualization\n",
        "\n",
        "This notebook visualizes the results of a CVAE trained on the MedMNIST dataset."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import torch\n",
        "import torchvision\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from model import CVAE\n",
        "\n",
        "# Load the trained CVAE model\n",
        "cvae = CVAE(img_channels=1, label_dim=10)\n",
        "cvae.load_state_dict(torch.load('cvae_medmnist.pth'))\n",
        "cvae.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\n",
        "cvae.eval()\n",
        "\n",
        "# Function to generate new images\n",
        "def generate_images(cvae, num_images=64, label=0):\n",
        "    with torch.no_grad():\n",
        "        # Create random latent vectors\n",
        "        z = torch.randn(num_images, 256).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\n",
        "        # Create one-hot encoded labels\n",
        "        y = torch.zeros(num_images, 10).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\n",
        "        y[:, label] = 1\n",
        "        # Decode the latent vectors\n",
        "        generated_images = cvae.decode(z, y).cpu()\n",
        "        return generated_images\n",
        "\n",
        "# Generate and visualize images for each label\n",
        "fig, axs = plt.subplots(10, 8, figsize=(15, 20))\n",
        "for label in range(10):\n",
        "    generated_images = generate_images(cvae, num_images=8, label=label)\n",
        "    for i, img in enumerate(generated_images):\n",
        "        axs[label, i].imshow(img.squeeze(0), cmap='gray')\n",
        "        axs[label, i].axis('off')\n",
        "\n",
        "plt.show()"
      ],
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
