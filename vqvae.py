import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_input = x.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, img_channels=1, img_size=28, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.embedding_dim = embedding_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(img_channels, 128, kernel_size=4, stride=2, padding=1)  # 28x28 -> 14x14
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7
        self.enc_conv3 = nn.Conv2d(256, embedding_dim, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(embedding_dim, 256, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.dec_conv3 = nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.enc_conv3(x)
        return x

    def decode(self, x):
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss, perplexity = self.vq(z)
        x_recon = self.decode(quantized)
        return x_recon, vq_loss, perplexity
