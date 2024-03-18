import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from ddpm import DDPM
from CNN import CNN

# Specify the MPS device
device = torch.device("mps")

# Preprocessing and set up data loader
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)

# Define train_size and val_size based on 80/20 split
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Now create your DataLoaders
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

# Initialise model with parameters
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)

# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)
ddpm = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000).to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

accelerator = Accelerator()

# We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
ddpm, optim, train_dataloader, val_dataloader = accelerator.prepare(
    ddpm, optim, train_dataloader, val_dataloader
)

# Set the number of epochs for training
n_epoch = 100
train_losses = []
val_losses = []

for i in range(n_epoch):
    ddpm.train()

    # Lists to store losses for each batch in the current epoch
    train_batch_losses = []
    val_batch_losses = []

    # Wrap the loop with a visual progress bar
    pbar = tqdm(train_dataloader)
    for x, _ in pbar:
        x = x.to(accelerator.device)
        optim.zero_grad()  # Clear old gradient from the previous step

        loss = ddpm(x)

        loss.backward()  # Backpropagation

        train_batch_losses.append(loss.item())

        # Calculates the running average of the last 100 batch losses or fewer if <100 training batches
        avg_loss = np.average(
            train_batch_losses[max(len(train_batch_losses) - 100, 0) :]
        )

        pbar.set_description(
            f"loss:{avg_loss:.3g}"
        )  # Show running average of loss in progress bar

        optim.step()

    # Save model and samples every 10 epochs

    ddpm.eval()
    with torch.no_grad():
        for x_val, _ in val_dataloader:
            x_val = x_val.to(accelerator.device)

            val_loss = ddpm(x_val)
            val_batch_losses.append(val_loss.item())

        # Generate and save samples after validation
        xh = ddpm.sample(16, (1, 28, 28), accelerator.device)
        grid = make_grid(xh, nrow=4)

        # Save samples to `./contents` directory
        save_image(grid, f"./contents/ddpm_sample_{i:04d}.png")

        # save model
        torch.save(ddpm.state_dict(), "./ddpm_mnist.pth")

    # Compute the average loss for the current epoch for both training and validation, then store it.
    train_epoch_loss = np.mean(train_batch_losses)
    val_epoch_loss = np.mean(val_batch_losses)
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)

    # Save model after each epoch
    torch.save(ddpm.state_dict(), "./ddpm_mnist.pth")

# Plotting training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training & Validation Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("./contents/loss_curves.png")
plt.show()
