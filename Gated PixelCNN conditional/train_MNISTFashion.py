import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from masked import MaskedConv2d
from condition import ConditionEncoder
from gated_pixelcnn import GatedPixelCNN
from fonctions_train import train_epoch, evaluate, main
from checkpoint import save_checkpoint, load_checkpoint
from sample import discretize_images, generate_samples
from visualize import visualize_samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du dispositif: {device}")


# Hyperparamètres
batch_size = 128
learning_rate = 5e-3
num_epochs = 20
n_classes = 10
in_channels = 1
hidden_channels = 64
n_layers = 10
output_dim = 32


# Préparation des données (Fashion MNIST)
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



# Modèle
model = GatedPixelCNN(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    n_layers=n_layers,
    n_classes=n_classes,
    output_dim=output_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":
    main()