import torch
import torch.nn as nn

class SMDANet(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(
            d_model,
            input_dim
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_projection(x)
        return x
