import matplotlib.pyplot as plt
import numpy as np


# Fonction pour visualiser les échantillons générés
def visualize_samples(samples, class_names=None):
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))

    if n_samples == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        img = sample.squeeze(0).numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        if class_names is not None and i < len(class_names):
            axes[i].set_title(class_names[i])

    plt.tight_layout()
    return fig