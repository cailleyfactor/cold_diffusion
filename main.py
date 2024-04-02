"""
@file main.py
@brief Main script for training and evaluating the DDPM model on the MNIST dataset.
@author Created by C. Factor on 10/03/2024 and involves code from a starter notebook
provided by Miles Cranmer for the coursework project.
"""
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from ddpm import DDPM
from CNN import CNN
from train import train_model
import os
import pandas as pd

export = 0.0

# Hyperparameters
n_epoch = 100
batch_size = 128
n_hidden = (16, 32, 32, 16)
act = nn.GELU
lr = 2e-4
betas = (1e-4, 0.02)
n_T = 1000

# Changed hyperparameter n_hidden = (16, 32, 64, 32, 16)

# Set the number of workers for data loader
num_workers = 0

# Specify the paths for saving
model_path = "./results/ddpm_mnist_default.pth"
sample_dir = "./results/results_default_model"
losses_path = "./results/losses_default_model.csv"

# Make sample directory if it does not exist
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Check if MPS is available and set the device accordingly
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

# Preprocessing and set up data loader
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)

# Define train_size and val_size based on 80/20 split
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

# Define train and validation datasets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# Initialise model, optim, accelerator
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=act)
ddpm = DDPM(gt=gt, betas=betas, n_T=n_T).to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
accelerator = Accelerator()

# We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
ddpm, optim, train_dataloader, val_dataloader = accelerator.prepare(
    ddpm, optim, train_dataloader, val_dataloader
)

train_losses, val_losses, ddpm = train_model(
    n_epoch,
    ddpm,
    optim,
    train_dataloader,
    val_dataloader,
    accelerator,
    model_path,
    sample_dir,
)

# Save trained model
torch.save(ddpm.state_dict(), model_path)

# Convert losses to Pandas DataFrame
df_losses = pd.DataFrame({"train_losses": train_losses, "val_losses": val_losses})

# Plotting training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training & Validation Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(sample_dir, "loss_curves.png"))
plt.show()
