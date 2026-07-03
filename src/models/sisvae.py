import torch
import torch.nn as nn

class SISVAE(nn.Module):

    def __init__(self, input_dim, latent_dim=16, hidden_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h, _ = self.encoder(x)
        h_last = h[:, -1, :]

        mu = self.mu_layer(h_last)
        logvar = self.logvar_layer(h_last)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        h, _ = self.decoder(z)
        return self.output_layer(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar
