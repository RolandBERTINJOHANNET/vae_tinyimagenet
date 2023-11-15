import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import lpips

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x), inplace=True)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.class_to_idx = {}
        for idx, class_name in enumerate(os.listdir(os.path.join(root_dir, 'train'))):
            self.class_to_idx[class_name] = idx

        if self.split == 'train':
            self.data_path = os.path.join(root_dir, 'train')
            self.image_paths = []
            self.labels = []

            for label_dir in os.listdir(self.data_path):
                for image_file in os.listdir(os.path.join(self.data_path, label_dir, 'images')):
                    self.image_paths.append(os.path.join(self.data_path, label_dir, 'images', image_file))
                    self.labels.append(self.class_to_idx[label_dir])

        elif self.split == 'val':
            self.data_path = os.path.join(root_dir, 'val')
            with open(os.path.join(self.data_path, 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
                self.image_paths = [os.path.join(self.data_path, 'images', line.split('\t')[0]) for line in lines]
                self.labels = [self.class_to_idx[line.split('\t')[1]] for line in lines]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Convert to RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = (image*2.)-1.
        return image, label



import torch
from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims=None, beta=1.0, upsampling='bilinear', loss_type='lpips'):
        super(VanillaVAE, self).__init__()
        self.lpips_model = lpips.LPIPS(net='vgg', lpips=False) if loss_type == 'lpips' else None
        self.latent_dim = latent_dim
        hidden_dims = hidden_dims or [64, 128, 256, 512]
        self.beta = beta
        self.upsampling = upsampling
        self.loss_type = loss_type

        # Encoder setup
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*4*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4*4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)

        modules = []
        for i in range(len(hidden_dims) - 1, -1, -1):  # Adjusted to go one iteration further
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i - 1] if i != 0 else hidden_dims[0]  # Adjust for the final layer
            modules.append(self._deconv_block(in_channels, out_channels))

        # Final convolution to adjust to the required output channels (e.g., 3 for RGB images), followed by Tanh
        modules.append(nn.Sequential(
            nn.Conv2d(hidden_dims[0], 3, kernel_size=3, padding=1),
            nn.Tanh()
        ))
        self.decoder = nn.Sequential(*modules)

    def _deconv_block(self, in_channels, out_channels):
        layers = []

        # Upsampling
        if self.upsampling == 'convtranspose':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        elif self.upsampling == 'pixelshuffle':
            layers.append(nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(upscale_factor=2))
        else:
            layers.append(nn.Upsample(scale_factor=2, mode=self.upsampling))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        # Final Activation
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())


        # Residual Block
        layers.append(ResidualBlock(out_channels))

        return nn.Sequential(*layers)


    def _final_layer(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, input):
        result = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_mu(result), self.fc_var(result)

    def decode(self, z):
        result = self.decoder_input(z).view(-1, 512, 4,4)
        return self.decoder(result)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.randn_like(std) * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, recons, input, mu, log_var, **kwargs):
        # Reconstruction Loss
        if self.loss_type == 'mse':
            recons_loss = F.mse_loss(recons, input)
        elif self.loss_type == 'lpips':
            recons_loss = self.lpips_model(recons, input).mean()

        # KLD Loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
        kld_loss = kld_loss.mean()

        # Final VAE Loss
        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}


    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]

import torch.optim as optim
from tqdm import tqdm
import os

from torch.optim.lr_scheduler import StepLR

def train_vae(model, train_loader, val_loader, device, n_epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    step_size = max(1, n_epochs // 2)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)

    batch_losses = []
    val_losses = []
    recons_losses = []
    kld_losses = []
    val_recons_losses = []
    val_kld_losses = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_recons_loss = 0
        train_kld_loss = 0

        for images, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, _, mu, log_var = model(images)
            loss_dict = model.loss_function(recon_images, images, mu, log_var, M_N=1./len(train_loader))
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_recons_loss += loss_dict['Reconstruction_Loss'].item()
            train_kld_loss += loss_dict['KLD'].item()

        avg_train_loss = train_loss / len(train_loader)
        batch_losses.append(avg_train_loss)
        recons_losses.append(train_recons_loss / len(train_loader))
        kld_losses.append(train_kld_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        val_recons_loss = 0
        val_kld_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon_images, _, mu, log_var = model(images)
                loss_dict = model.loss_function(recon_images, images, mu, log_var, M_N=1./len(val_loader))
                val_loss += loss_dict['loss'].item()
                val_recons_loss += loss_dict['Reconstruction_Loss'].item()
                val_kld_loss += loss_dict['KLD'].item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_recons_losses.append(val_recons_loss / len(val_loader))
        val_kld_losses.append(val_kld_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step()

    return batch_losses, val_losses, recons_losses, kld_losses, val_recons_losses, val_kld_losses



import matplotlib.pyplot as plt

def reconstruct_and_save(model, data_loader, device, num_images=5, filename='reconstructions.png'):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients here
        # Take a single batch from the data loader
        images, _ = next(iter(data_loader))
        # Just use the first `num_images` from the batch
        images = images[:num_images].to(device)
        reconstructions, _, _, _ = model(images)

        # Convert the images and reconstructions to numpy arrays
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        reconstructions = reconstructions.cpu().numpy().transpose(0, 2, 3, 1)

        fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(15, 5))
        for i in range(num_images):
            axes[0, i].imshow((images[i] * 0.5) + 0.5)  # Denormalize image
            axes[0, i].axis('off')
            axes[0, i].set_title('Original')
            axes[1, i].imshow((reconstructions[i] * 0.5) + 0.5)  # Denormalize reconstruction
            axes[1, i].axis('off')
            axes[1, i].set_title('Reconstruction')
        plt.tight_layout()
        plt.savefig(filename)  # Save the figure
        plt.close(fig)  # Close the figure to free up memory


import numpy as np
import matplotlib.pyplot as plt

def denormalize(tensor, mean, std):
    """
    Denormalizes the image tensor from given mean and std.
    """
    if len(tensor.shape) == 3:  # If tensor is 3D, add a batch dimension
        tensor = tensor.unsqueeze(0)

    for i in range(tensor.size(1)):  # Now tensor shape should be [Batch, Channels, Height, Width]
        tensor[:, i] = tensor[:, i] * std[i] + mean[i]

    if tensor.shape[0] == 1:  # If there was only one image, remove the batch dimension
        tensor = tensor.squeeze(0)

    return tensor


def interpolate_and_save(model, test_loader, device, n_interpolations=5, filename='interpolations.png'):
    model.eval()

    # Get a pair of images from the test_loader
    image_iterator = iter(test_loader)
    images1, _ = next(image_iterator)
    images2, _ = next(image_iterator)

    fig, axes = plt.subplots(nrows=5, ncols=n_interpolations+2, figsize=(15, 10))

    with torch.no_grad():
        for row in range(5):
            # Encode images
            image1 = images1[row].unsqueeze(0).to(device)
            image2 = images2[row].unsqueeze(0).to(device)

            mu1, log_var1 = model.encode(image1)
            mu2, log_var2 = model.encode(image2)

            z1 = model.reparameterize(mu1, log_var1)
            z2 = model.reparameterize(mu2, log_var2)

            # Interpolation
            for col, alpha in enumerate(np.linspace(0, 1, n_interpolations+2)):
                z = alpha * z1 + (1 - alpha) * z2
                interpolated_image = model.decode(z).cpu().squeeze(0)
                # Denormalize using the simpler method
                interpolated_image_denorm = (interpolated_image * 0.5) + 0.5
                axes[row, col].imshow(np.transpose(interpolated_image_denorm.numpy(), (1, 2, 0)))
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the figure with a high resolution
    plt.close(fig)  # Close the plot to free up memory


def interpolate_2x2_matrix(model, test_loader, device, n_interpolations=5, filename='matrix_interpolation.png'):
    model.eval()

    # Randomly select four images from the test_loader to serve as corner images
    image_iterator = iter(test_loader)
    images = next(image_iterator)[0][:4]  # Take the first four images from a batch

    assert len(images) == 4, "Ensure the test_loader batch size is at least 4."

    fig, axes = plt.subplots(nrows=n_interpolations+2, ncols=n_interpolations+2, figsize=(10, 10))

    with torch.no_grad():
        # Encode the corner images to get their latent representations
        z_corners = [model.encode(img.unsqueeze(0).to(device))[0] for img in images]  # [z_A, z_B, z_C, z_D]

        # Interpolate in the latent space
        for i, alpha in enumerate(np.linspace(0, 1, n_interpolations+2)):
            for j, beta in enumerate(np.linspace(0, 1, n_interpolations+2)):
                z = (1-alpha)*(1-beta)*z_corners[0] + alpha*(1-beta)*z_corners[1] + (1-alpha)*beta*z_corners[2] + alpha*beta*z_corners[3]
                interpolated_image = model.decode(z).cpu().squeeze(0)
                # Denormalize using the simpler method
                interpolated_image_denorm = (interpolated_image * 0.5) + 0.5
                axes[i, j].imshow(np.transpose(interpolated_image_denorm.numpy(), (1, 2, 0)))
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the figure with a high resolution
    plt.close(fig)  # Close the plot to free up memory


import os
import csv

def save_training_results(model, batch_losses, val_losses, recons_losses, kld_losses, val_recons_losses, val_kld_losses, upsampling_method, loss_function, latent_dim, root_dir):
    model_dir = os.path.join(root_dir, f"{upsampling_method}_{loss_function}_latent_dim={latent_dim}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Save losses to a CSV file
    loss_file_path = os.path.join(model_dir, 'losses.csv')
    with open(loss_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Train Reconstruction Loss", "Train KLD Loss", "Validation Reconstruction Loss", "Validation KLD Loss"])
        for epoch, data in enumerate(zip(batch_losses, val_losses, recons_losses, kld_losses, val_recons_losses, val_kld_losses)):
            writer.writerow([epoch + 1] + list(data))
    # Generate and save reconstructions
    reconstruct_and_save(model, val_loader, device, num_images=5, filename=os.path.join(model_dir, 'reconstructions.png'))

    # Generate and save interpolations
    interpolate_and_save(model, val_loader, device, n_interpolations=5, filename=os.path.join(model_dir, 'interpolations.png'))

    # Generate and save 2x2 matrix interpolations (if you have this function)
    interpolate_2x2_matrix(model, val_loader, device, n_interpolations=5, filename=os.path.join(model_dir, 'matrix_interpolation.png'))



################################################CREATE DATASET
# Set the root directory where TinyImageNet is located
DATASET_DIR = os.environ.get('DATASET_DIR', '.')  # Get the environment variable, if not set, default to current directory
root_dir = os.path.join(DATASET_DIR, "tiny-imagenet-200")

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets
train_dataset = TinyImageNetDataset(root_dir, split='train', transform=transform)
val_dataset = TinyImageNetDataset(root_dir, split='val', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print("images loaded")

######################################################check inputs distribution
print("checking out inputs distribution : \n\n")
data, _ = next(iter(train_loader))
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean(dim=[0, 2, 3]))
print("Std:", data.std(dim=[0, 2, 3]))

upsampling_methods = ['pixelshuffle', 'nearest', 'bilinear', 'convtranspose']
loss_functions = ['lpips', 'mse']
latent_dims = [250]
root_dir = 'resid_results_bigexpe_tanh'

for latent_dim in latent_dims:
    for upsampling_method in upsampling_methods:
        for loss_function in loss_functions:
            model = VanillaVAE(in_channels=3, latent_dim=latent_dim, beta=0.0005,
                               upsampling=upsampling_method, loss_type=loss_function).to(device)
            print(f"\nTraining model with latent dimension: {latent_dim}, upsampling: {upsampling_method}, loss: {loss_function}")

            # Train the model
            batch_losses, val_losses, recons_losses, kld_losses, val_recons_losses, val_kld_losses = train_vae(model, train_loader, val_loader, device, n_epochs=50)

            # Save results
            save_training_results(model, batch_losses, val_losses, recons_losses, kld_losses, val_recons_losses, val_kld_losses, upsampling_method, loss_function, latent_dim, root_dir)
