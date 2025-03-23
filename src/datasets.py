import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader, TensorDataset, Subset

from pathlib import Path

datasets_folder = Path(__file__).parent.parent / "datasets"
datasets_folder.mkdir(exist_ok=True)


def preprocess(data, device):
    """ Load data on the correct device and return a TensorDataset
        than can later be used with a DataLoader
    """
    inputs, labels = map(list, zip(*data))
    inputs = torch.stack(inputs).to(device)
    labels = torch.tensor(labels, dtype=torch.uint8).to(device)

    return TensorDataset(inputs, labels)

def mnist(device):

    train_data = datasets.MNIST(str(datasets_folder),download=True,train=True, transform=ToTensor())
    test_data  = datasets.MNIST(str(datasets_folder),download=True,train=False, transform=ToTensor())

    return preprocess(train_data, device), preprocess(test_data, device)

def fashion_mnist(device):

    train_data = datasets.FashionMNIST(str(datasets_folder),download=True,train=True, transform=ToTensor())
    test_data  = datasets.FashionMNIST(str(datasets_folder),download=True,train=False, transform=ToTensor())

    return preprocess(train_data, device), preprocess(test_data, device)

def cifar_10(device):

    train_data = datasets.CIFAR10(str(datasets_folder),download=True,train=True, transform=ToTensor())
    test_data  = datasets.CIFAR10(str(datasets_folder),download=True,train=False, transform=ToTensor())

    return preprocess(train_data, device), preprocess(test_data, device)
