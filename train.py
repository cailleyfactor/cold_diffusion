"""
!@file train.py
@brief Module containing tool for training the DDPM model on the MNIST dataset.
@author Created by C. Factor on 10/03/2024 and involves code from a starter notebook
provided by Miles Cranmer for the coursework project.
"""
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import os


def train_model(
    n_epoch,
    ddpm,
    optim,
    train_dataloader,
    val_dataloader,
    accelerator,
    model_path,
    sample_dir,
):
    """
    @brief Function to train the DDPM model.
    @param n_epoch: The number of epochs to train the model.
    @param ddpm: The DDPM model.
    @param optim: The optimizer.
    @param train_dataloader: The DataLoader for the training dataset.
    @param val_dataloader: The DataLoader for the validation dataset.
    @param accelerator: The Accelerator object.
    @param model_path: The path to save the trained model.
    @param sample_dir: The directory to save the samples.
    @return train_losses: The training losses.
    @return val_losses: The validation losses.
    @return ddpm: The trained DDPM model."""
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

            loss = ddpm(x)  # Call the forward method of DDPM model

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

        # Evaluate the model on the validation dataset and generate samples
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
            save_image(grid, os.path.join(sample_dir, f"ddpm_sample_{i:04d}.png"))

            # save model
            torch.save(ddpm.state_dict(), model_path)

        # Compute the average loss for the current epoch for both training and validation, then store it.
        train_epoch_loss = np.mean(train_batch_losses)
        val_epoch_loss = np.mean(val_batch_losses)
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

    return train_losses, val_losses, ddpm