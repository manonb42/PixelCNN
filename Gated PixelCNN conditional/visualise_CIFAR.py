import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_samples(samples, class_names=None):
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
    if n_samples == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        img = sample.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        if class_names is not None and i < len(class_names):
            axes[i].set_title(class_names[i])

    plt.tight_layout()
    return fig


 
def generate_samples(model, class_labels, num_samples=10, shape=(3, 32, 32), device='cpu'):
    samples = []
    model.eval()

    for label in class_labels:
        condition = torch.tensor([label], device=device)
        print(f"Génération d'un échantillon pour la classe {label}...")
        sample = model.sample(condition=condition, shape=shape, device=device)
        samples.append(sample.cpu().squeeze(0))

    return samples