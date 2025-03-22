import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from masked import MaskedConv2d
from condition import ConditionEncoder


class GatedActivation(nn.Module):
    """
    Implémentation de l'activation gated
    """
    def forward(self, x):
        x_tanh, x_sigmoid = torch.chunk(x, 2, dim=1)
        return torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)


class GatedPixelCNNLayer(nn.Module):
    """
    Couche de base du Gated PixelCNN avec connections verticales et horizontales.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_type='B'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Connexion verticale
        self.v_conv = MaskedConv2d(in_channels, 2*out_channels,
                                  kernel_size=(kernel_size, 1),
                                  mask_type=mask_type)

        # Connexion horizontale
        self.h_conv = MaskedConv2d(in_channels, 2*out_channels,
                                  kernel_size=(1, kernel_size),
                                  mask_type=mask_type)

        self.v_to_h = nn.Conv2d(2*out_channels, 2*out_channels, kernel_size=1)
        self.h_skip = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.gated_activation = GatedActivation()

        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.residual_conn = None
        if in_channels != out_channels:
            self.residual_conn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        v_out = self.v_conv(x)
        h_out = self.h_conv(x)

        # Ajout de l'information verticale à l'horizontale
        v_to_h = self.v_to_h(v_out)
        h_out = h_out + v_to_h

        h_out = self.gated_activation(h_out)  

        skip = self.h_skip(h_out)
        h_out = self.out_conv(h_out)

        if self.residual_conn is not None:
            x = self.residual_conn(x)
        h_out = h_out + x  
        return h_out, skip




class GatedPixelCNN(nn.Module):
    """
    Implémentation du modèle Gated PixelCNN conditionnel.
    """
    def __init__(self, in_channels=3, hidden_channels=64, n_layers=15,
                 kernel_size=7, n_classes=10, output_dim=256):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.in_channels = in_channels

        # Couche d'entrée - Masked Conv de type A
        self.input_conv = MaskedConv2d(in_channels, hidden_channels,
                                     kernel_size=kernel_size, mask_type='A')

        # Encodeur de condition
        self.cond_encoder = ConditionEncoder(n_classes, embedding_dim=64,
                                            hidden_dim=128, output_dim=hidden_channels)

        # Couches Gated PixelCNN
        self.gated_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gated_layers.append(
                GatedPixelCNNLayer(hidden_channels, hidden_channels, kernel_size=3)
            )

        # Couches de sortie
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_dim * in_channels, kernel_size=1)
        )

    def forward(self, x, condition=None):
        batch_size, in_channels, height, width = x.shape
        assert in_channels == self.in_channels, f"Expected {self.in_channels} input channels, got {in_channels}"

        # Transformation conditionnelle
        cond_features = None
        if condition is not None:
            cond_vector = self.cond_encoder(condition)
            cond_features = cond_vector.view(batch_size, -1, 1, 1)
            cond_features = cond_features.expand(-1, -1, height, width)

        x = self.input_conv(x)
        if cond_features is not None:
            x = x + cond_features

        skip_connections = torch.zeros_like(x)

        for layer in self.gated_layers:
            x, skip = layer(x)
            skip_connections = skip_connections + skip

        out = self.output_layers(skip_connections)

        out = out.view(batch_size, in_channels, self.output_dim, height, width)
        out = out.permute(0, 1, 3, 4, 2)
        return out

    def sample(self, condition=None, shape=(3, 32, 32), device='cpu'):
        """
        Génère un échantillon à partir du modèle en échantillonnant pixel par pixel.
        """
        with torch.no_grad():
            sample = torch.zeros(1, *shape, device=device)
            for h in range(shape[1]):
                for w in range(shape[2]):
                    for c in range(shape[0]):
                        out = self.forward(sample, condition)
                        probs = F.softmax(out[0, c, h, w], dim=0)
                        sample[0, c, h, w] = torch.multinomial(probs, 1).float() / self.output_dim

            return sample