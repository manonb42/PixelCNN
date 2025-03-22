import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConditionEncoder(nn.Module):
    """
    Encoder l'information conditionnelle en un vecteur d'embedding
    """
    def __init__(self, num_classes, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, condition):
        x = self.embedding(condition)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x