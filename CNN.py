"""
@file CNN.py
@brief Module containing the convolutional neural network (CNN) model for the
denoising diffusion probabilistic model (DDPM).
@details Initialises the CNN model, defines the forward pass, and provides a sampling function.

"""
import numpy as np
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    @class CNNBlock
    @brief Class for the convolutional neural network (CNN) block.
    @details Defines a basic building block of the CNN, which is used multiple times in the main CNN model.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        """
        @brief Initialises the CNN block.
        @param in_channels: The number of input channels.
        @param out_channels: The number of output channels.
        @param expected_shape: The expected shape of the input tensor.
        @param act: The activation function.
        @param kernel_size: The kernel size for the convolutional layer.
        @return None
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act(),
        )

    def forward(self, x):
        """
        @brief Implements the forward method of the CNN block.
        @param x: The input tensor.
        @return The output tensor."""
        return self.net(x)


class CNN(nn.Module):
    """
    @class CNN
    @brief Class for the convolutional neural network (CNN) model for the
    denoising diffusion probabilistic model (DDPM).
    @details This class initialises the CNN model. It integrates time information into the model processing images
    through a series of convolutional blocks.
    """

    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        """
        @brief Initialises the CNN model.
        @details  Initializes the CNN model, setting up the architecture including the convolutional blocks,
        final convolutional layer, and time embedding layers.
        @param in_channels: The number of input channels.
        @param expected_shape: The expected shape of the input tensor.
        @param n_hidden: The number of hidden channels.
        @param kernel_size: The kernel size for the convolutional layers.
        @param last_kernel_size: The kernel size for the final convolutional layer.
        @param time_embeddings: The number of time embeddings.
        @param act: The activation function.
        @return None"""
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        # This part is literally just to put the single scalar "t" into the CNN
        # in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        @brief Encodes the time information into the latent space.
        @details Encodes a given time tensor t into a high-dimensional space
        using sine and cosine functions, followed by linear transformations.
        @param t: The time tensor.
        @return The time encoding tensor."""
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        @brief Implements the forward method of the CNN model.
        @details Implements the forward pass of the CNN model,
        integrating both spatial and temporal information to produce the final output.
        @param x: The input tensor.
        @param t: The time tensor.
        @return The output tensor."""
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed
