import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# Fonction pour discrétiser les images
def discretize_images(images, output_dim=256):
    return (images * (output_dim - 1)).long()


# Fonction pour générer des échantillons conditionnels
def generate_samples(model, class_labels, num_samples=10, shape=(1, 28, 28), device='cpu'):
    samples = []
    model.eval()
    for label in class_labels:
        condition = torch.tensor([label], device=device)

        print(f"Génération d'un échantillon pour la classe {label}...")
        sample = model.sample(condition=condition, shape=shape, device=device)
        samples.append(sample.cpu().squeeze(0))
    return samples