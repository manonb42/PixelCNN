import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader, TensorDataset, Subset

from pathlib import Path

datasets_folder = Path(__file__).parent.parent / "datasets"
datasets_folder.mkdir(exist_ok=True)

def mnist(device, binarize = False):

    binarize = Lambda(lambda x: x > 0.5)
    transform = Compose([ToTensor(), binarize]) if binarize else ToTensor()

    train_data = datasets.MNIST(".",download=True,train=True, transform=transform)
    test_data  = datasets.MNIST(".",download=True,train=False, transform=transform)

    inputs, labels = map(list, zip(*train_data))
    inputs = torch.stack(inputs).to(device)
    labels = torch.tensor(labels, dtype=torch.uint8).to(device)

    train = TensorDataset(inputs, labels)


    inputs, labels = map(list, zip(*test_data))
    inputs = torch.stack(inputs).to(device)
    labels = torch.tensor(labels, dtype=torch.uint8).to(device)

    test = TensorDataset(inputs, labels)

    return train, test
