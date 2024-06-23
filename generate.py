with torch.no_grad():
    sample = torch.randn(64, latent_dim).to(device)
    generated_images = vae.decode(sample).cpu()
    grid_img = torchvision.utils.make_grid(generated_images.view(64, 1, 28, 28), nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.show()
