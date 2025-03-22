import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sample import discretize_images, generate_samples
from checkpoint import save_checkpoint, load_checkpoint
from visualise_CIFAR import visualize_samples, generate_samples
from fonctions_train import train_epoch, evaluate


def main():
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
                  'chien', 'grenouille', 'cheval', 'bateau', 'camion']

    checkpoint_path = os.path.join(checkpoint_dir, 'pixelcnn_latest.pth')
    start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)

    train_losses = []
    val_losses = []

    # Boucle d'entraînement
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, device, output_dim)
        train_losses.append(train_loss)

        val_loss = evaluate(model, test_loader, criterion, device, output_dim)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch + 1, val_loss,
                       os.path.join(checkpoint_dir, 'pixelcnn_latest.pth'))

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss,
                           os.path.join(checkpoint_dir, 'pixelcnn_best.pth'))

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            selected_classes = list(range(n_classes))
            samples = generate_samples(model, selected_classes,
                                      shape=(in_channels, 32, 32), device=device)

            fig = visualize_samples(samples, [class_names[c] for c in selected_classes])
            fig.savefig(f'samples_epoch_{epoch+1}.png')
            plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_history.png')
    plt.close()

    print("Entraînement terminé!")

    best_model_path = os.path.join(checkpoint_dir, 'pixelcnn_best.pth')
    load_checkpoint(model, optimizer, best_model_path)

    final_samples = generate_samples(model, range(n_classes),
                                    shape=(in_channels, 32, 32), device=device)

    fig = visualize_samples(final_samples, class_names)
    fig.savefig('final_samples.png')
    plt.close(fig)