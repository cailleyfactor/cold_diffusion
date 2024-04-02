"""@file ddpm_schedules.py
@brief Module containing functions for creating a DDPM training schedule.
@author Created by C. Factor on 10/03/2024 and
involves code from a starter notebook provided by Miles Cranmer for the coursework project.
"""
from typing import Dict
import torch


# Create a DDPM training schedule for use when evaluating and training the diffusion model
def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    @brief Returns pre-computed schedules for DDPM sampling with a linear noise schedule.
    @param beta1: The initial beta value.
    @param beta2: The final beta value.
    @param T: The number of timesteps.
    @return A dictionary containing the beta and alpha values with a linear noise schedule.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}
