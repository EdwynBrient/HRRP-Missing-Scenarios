import torch
from math import pi
import torch.nn as nn


class CosineScheduler(nn.Module):
    def __init__(self, T: int, s: float = 0.008, clipping_value: float = 0.999):
        super().__init__()
        """
        Cosine variance scheduler.
        alpha_hat = min(cos((t / T + s) / (1 + s) * pi / 2)^2, clipping_value)
        beta = 1 - (alpha_hat(t) / alpha_hat(t - 1))
        beta_hat = (1 - alpha_hat(t - 1)) / (1 - alpha_hat(t)) * beta(t)
        """
        self.T = T
        self.clipping_value = torch.Tensor([clipping_value])
        self._alpha_hats = self._alpha_hat_function(torch.arange(self.T), T, s)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, shifts=1, dims=0)
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]  # remove first NaN value
        self._betas = 1.0 - self._alpha_hats / self._alpha_hats_t_minus_1
        self._betas = torch.minimum(self._betas, self.clipping_value)
        self._alphas = 1.0 - self._betas
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas
        self._betas_hat[torch.isnan(self._betas_hat)] = 0.0

    def forward(self):
        return self._betas

    def _alpha_hat_function(self, t: torch.Tensor, T: int, s: float):
        cos_value = torch.pow(torch.cos((t / T + s) / (1 + s) * pi / 2.0), 2)
        return cos_value

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat
