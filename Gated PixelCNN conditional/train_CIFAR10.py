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
from sample import discretize_images
from checkpoint import save_checkpoint, load_checkpoint
from visualise_CIFAR import visualize_samples, generate_samples
from fonctions_train import train_epoch, evaluate
from main_CIFAR import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du dispositif: {device}")


batch_size = 128
learning_rate = 3e-3
num_epochs = 25
n_classes = 10
in_channels = 3
hidden_channels = 128
n_layers = 10
output_dim = 256 


transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model = GatedPixelCNN(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    n_layers=n_layers,
    n_classes=n_classes,
    output_dim=output_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":
    main()
