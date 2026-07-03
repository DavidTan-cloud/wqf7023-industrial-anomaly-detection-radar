import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class RANCoder:
    def __init__(self,
                 input_dim,
                 hidden_dims=[32, 64, 128],
                 device="cpu"):

        self.device = device

        self.models = [
            SimpleAutoencoder(
                input_dim,
                h
            ).to(device)

            for h in hidden_dims
        ]

    def fit(self,
            X,
            epochs=20,
            lr=0.001):

        criterion = nn.MSELoss()

        for i, model in enumerate(self.models):

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr
            )

            for epoch in range(epochs):

                optimizer.zero_grad()

                output = model(X)

                loss = criterion(
                    output,
                    X
                )

                loss.backward()
                optimizer.step()

            print(f"AE {i+1} trained")

    def score(self, X):
        errors = []
        with torch.no_grad():
            for model in self.models:
                output = model(X)
                err = ((X - output) ** 2).mean(dim=1)
                errors.append(err.cpu().numpy())
        return sum(errors) / len(errors)
