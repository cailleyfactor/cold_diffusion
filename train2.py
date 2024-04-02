"""!@file ddpm2.py
@brief Module containing tool for training, evaluating, and sampling from the DDPM model.
@details
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

        # Save model and samples every 10 epochs

        ddpm.eval()
        with torch.no_grad():
            x_starts = []
            n_sample = 16
            for x_val, _ in val_dataloader:
                x_val = x_val.to(accelerator.device)
                # Compute the loss for the validation set
                val_loss = ddpm(x_val)
                # Store the loss for the current batch
                val_batch_losses.append(val_loss.item())
                # Collect samples for validation
                if len(x_starts) < n_sample:
                    # Collect the first `n_sample` samples from the validation set
                    for single_x in x_val:
                        # Unsqueeze to maintain the batch dimension
                        x_starts.append(single_x)
                        # Break early if we've collected enough samples
                        if len(x_starts) >= n_sample:
                            break

            if len(x_starts) == n_sample:
                # Stack the collected samples along a new dimension
                x_start_stacked = torch.stack(x_starts, dim=0)
            # Generate and save samples after validation
            blur, xh, recon, z_0 = ddpm.sample(
                n_sample, (1, 28, 28), x_start_stacked, accelerator.device
            )
            grid = make_grid(xh, nrow=4)
            grid2 = make_grid(recon, nrow=4)
            grid3 = make_grid(z_0, nrow=4)
            grid6 = make_grid(blur, nrow=4)

            # Save samples to `./contents` directory
            save_image(grid, os.path.join(sample_dir, f"ddpm_output_algo2_{i:04d}.png"))
            save_image(grid2, os.path.join(sample_dir, f"ddpm_recon_{i:04d}.png"))
            save_image(grid3, os.path.join(sample_dir, f"ddpm_z_0_{i:04d}.png"))
            save_image(grid6, os.path.join(sample_dir, f"ddpm_blur_{i:04d}.png"))
            # save model
            torch.save(ddpm.state_dict(), model_path)

        # Compute the average loss for the current epoch for both training and validation, then store it.
        train_epoch_loss = np.mean(train_batch_losses)
        val_epoch_loss = np.mean(val_batch_losses)
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

    return train_losses, val_losses, ddpm
