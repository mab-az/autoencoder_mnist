import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 20
learning_rate = 1e-3

# https://afagarap.works/2020/01/26/implementing-autoencoder-in-pytorch.html


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"],
            out_features=128
            )
        self.encoder_output_layer = nn.Linear(
            in_features=128,
            out_features=128
        )
        self.encoder_hidden_layer = nn.Linear(
            in_features=128,
            out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128,
            out_features=kwargs["input_shape"]
        )


def forward(self, features):
    activation = self.encoder_hidden_layer(features)
    activation = torch.relu(activation)
    code = self.encoder_output_layer(activation)
    code = torch.relu(code)
    activation = self.decoder_hidden_layer(code)
    activation = torch.relu(activation)
    activation = self.decoder_output_layer(activation)
    reconstructed = torch.relu(activation)

    return reconstructed


device = torch.device("cpu")

model = AutoEncoder(input_shape=784).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)

for epoch in range(epochs):
    loss=0
    for batch_features, _ in train_loader:
        batch_features = batch_features.view(-1, 784).to(device)

        optimizer.zero_grad()
        outputs = model(batch_features)

        train_loss = criterion(outputs, batch_features)
        train_loss.backward()

        optimizer.step()

        loss += train_loss.item()

    loss = loss / len(train_loader)

    print(f"epoch: {epoch+1}/{epochs}, loss = {loss:.6f}")
