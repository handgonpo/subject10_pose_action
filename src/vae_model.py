from __future__ import annotations

import torch
from torch import nn


class SequenceVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()

        if hidden_dim < 2:
            raise ValueError(
                "hidden_dim은 2 이상이어야 합니다."
            )

        encoded_dim = hidden_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(
            encoded_dim,
            latent_dim,
        )
        self.logvar_layer = nn.Linear(
            encoded_dim,
            latent_dim,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoded_dim),
            nn.ReLU(),
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(
    reconstruction,
    x,
    mu,
    logvar,
    beta: float,
):
    reconstruction_loss = torch.mean(
        (reconstruction - x) ** 2
    )

    kl_loss = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = (
        reconstruction_loss
        + float(beta) * kl_loss
    )

    return total_loss, reconstruction_loss, kl_loss