"""!@file ddpm2.py
@brief Module containing functions for the denoising diffusion probabilistic model (DDPM),
modified for cold diffusion with Gaussian blurring.
@details Initialises the model, defines the forward pass,
and provides a sampling function.
@author Created by C. Factor on 10/03/2024 and involves code
from a starter notebook provided by Miles Cranmer for the coursework project and
is informed by code from the Bansal et al. paper and
repository available at: https://github.com/arpitbansal297/Cold-Diffusion-Models.
"""
from typing import Tuple
import torch
import torch.nn as nn
from ddpm_schedules import ddpm_schedules
import torchgeometry as tgm


class DDPM(nn.Module):
    """
    !@class DDPM
    @brief Class for the denoising diffusion probabilistic model (DDPM) with Gaussian blurring.
    @details This class is a modified version of the DDPM model
    that incorporates Gaussian blurring as a preprocessing step.
    """

    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """
        @brief Initialises the DDPM model.
        @param gt: The generative transformer model.
        @param betas: The beta values for the noise schedule.
        @param n_T: The number of timesteps.
        @param criterion: The loss function.
        @return None"""
        super().__init__()

        self.gt = gt
        # kernel_std = 7 and kernel_size = 13 for running the model with more blurring
        self.kernel_std = 0.1
        self.kernel_size = 3
        self.channels = 1
        # This line creates a noise schedule based on the beta values and number of timesteps
        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)
        self.n_T = n_T

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t
        self.gaussian_kernels = nn.ModuleList(self.get_kernels())
        self.n_T = n_T
        self.criterion = criterion

    # This function was based on the bansal paper & involves modified code from
    # `deblurring-diffusion-pytorch` from the Bansal repository
    def get_conv(self, dims, std, mode="circular"):
        """
        @brief Initialises a 2D convolutional layer with a Gaussian kernel.
        @param dims: The dimensions of the Gaussian kernel.
        @param std: The standard deviation of the Gaussian kernel.
        @param mode: The padding mode for the convolutional layer.
        @return conv: The convolutional layer with the Gaussian kernel."""
        # This line creates a Gaussian kernel with the specified dimensions and standard deviation
        kernel = tgm.image.get_gaussian_kernel2d(dims, std)
        # Initialises a 2D convolutional layer with the specified dimensions
        # and kernel size set to dimensions of Gaussian kernel
        # The padding is set to half the kernel size to ensure the output has the same dimensions as the input
        # The bias is set to False to prevent the model from learning the Gaussian kernel
        # The groups parameter is set to the number of channels to apply the same kernel to each channel
        conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=dims,
            padding=int((dims[0] - 1) / 2),
            padding_mode=mode,
            bias=False,
            groups=self.channels,
        )
        # Prepares the Gaussian kernel for use in a convolutional layer
        with torch.no_grad():
            # Unsqueezes (adds dimensions) twice to transform the kernel into a 4D tensor
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            # Repeats the kernel across the number of channels to match the input tensor
            kernel = kernel.repeat(self.channels, 1, 1, 1)
            # Sets the weights of the convolutional layer to the Gaussian kernel
            conv.weight = nn.Parameter(kernel)
        return conv

    # This function was based on the bansal paper
    # & involves modified code from `deblurring-diffusion-pytorch` from the Bansal repository
    def get_kernels(self):
        """
        @brief Initialises a list of convolutional layers configured as Gaussian kernels.
        @return kernels: The list of convolutional layers configured as Gaussian kernels.
        """
        # Initialises an empty list to store the convolutional layers configured as Gaussian kernels
        kernels = []
        # Iterates self.nT to create a list of Gaussian kernels with different changes with each time step
        for i in range(self.n_T):
            # This is what you need to change in accordance with the noise scheduler
            kernels.append(
                self.get_conv(
                    (self.kernel_size, self.kernel_size),
                    (self.kernel_std * (i + 1), self.kernel_std * (i + 1)),
                )
            )
        return kernels

    # This function was based on the bansal paper & informed by code from the Bansal repository
    def blur(self, x_start, t):
        """
        @brief Applies Gaussian blurring to the input tensor.
        @param x_start: The input tensor.
        @param t: The timestep.
        @return blurred: The blurred output tensor."""
        # Don't use gradients for this
        with torch.no_grad():
            # Initialize tensor for blurred outputs
            blurred = torch.zeros_like(x_start)
            # Iterate over each item in the batch and apply the corresponding Gaussian kernel
            for i in range(x_start.shape[0]):
                # Assuming t contains values starting from 1, we need to subtract 1 to get the correct index
                timestep = int(t[i].item() - 1)
                blurred[i] = self.gaussian_kernels[timestep](x_start[i])
        return blurred

    # This function was based on the bansal paper
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Implements the forward pass of the DDPM model.
        @param x: The input tensor.
        @return loss: The loss value.
        """
        # Randomly sample a timestep t from the range [1, n_T) of the size of the batch
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        # Apply the blur operation to the input tensor x
        x_blur = self.blur(x_start=x, t=t)
        # Reconstruct the original image using the CNN
        x_recon = self.gt(x_blur, t / self.n_T)
        return self.criterion(x, x_recon)

    # This function was based on the bansal paper (algorithm 2)
    def sample(
        self, n_sample: int, size, x_start_stacked: torch.Tensor, device
    ) -> torch.Tensor:
        """@brief Implements the sampling function for the DDPM model.
        @details This is an implementation of algorithm 2 described in the Bansal et al. paper.
        @param n_sample: The number of samples to generate.
        @param size: The size of the samples.
        @param x_start_stacked: The input tensor for the samples.
        @param device: The device to run the model on.
        @return z_t: The generated samples.
        @return recon: The reconstructed image from the decoder (CNN).
        """
        # Set the model to evaluation mode
        self.gt.eval()
        _one = torch.ones(n_sample, device=device)
        # Pull in a stacked batch of images from the validation set
        z_0 = x_start_stacked
        _one = torch.ones(n_sample, device=device)
        # Apply the blur operation to the input tensor x_start_stacked
        z_t = self.blur(x_start=z_0, t=(_one * self.n_T))
        # Iterate over each timestep in reverse order
        for i in range(self.n_T, 0, -1):
            # Reconstruct the original image using the CNN
            recon = self.gt(z_t, (i / self.n_T) * _one)
            # Apply the blur operation to the reconstructed image
            part_1 = self.blur(x_start=recon, t=i * _one)
            # Apply the blur operation to the reconstruction image but based on one less timestep
            part_2 = self.blur(x_start=recon, t=(i - 1) * _one)
            # Update the blurred image based on the reconstructed image and the Gaussian kernels
            if i > 1:
                # Last line of loop:
                z_t = z_t - part_1 + part_2
            # If the timestep is 1, set the blurred image to the reconstructed image
            else:
                z_t = z_t
        # Return the sample (confusingly named z_t) and the reconstructed image (recon)
        return z_t, recon
