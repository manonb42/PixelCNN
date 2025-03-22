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
from visualize import visualize_samples

# Fonction d'entraînement
def train_epoch(model, train_loader, optimizer, criterion, epoch, device, output_dim=256):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for i, (images, labels) in enumerate(pbar):

            images = images.to(device)
            labels = labels.to(device)

            images_discrete = discretize_images(images, output_dim)

            outputs = model(images, labels)
            loss = 0
            batch_size, in_channels, height, width, _ = outputs.shape

            for c in range(in_channels):
                loss += criterion(
                    outputs[:, c].contiguous().view(-1, output_dim),
                    images_discrete[:, c].contiguous().view(-1)
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})
    return running_loss / total_batches



# Fonction d'évaluation
def evaluate(model, test_loader, criterion, device, output_dim=256):
    model.eval()
    total_loss = 0.0
    total_batches = len(test_loader)

    with torch.no_grad():
        with tqdm(test_loader, desc="Évaluation") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                images_discrete = discretize_images(images, output_dim)
                outputs = model(images, labels)

                loss = 0
                batch_size, in_channels, height, width, _ = outputs.shape

                for c in range(in_channels):
                    loss += criterion(
                        outputs[:, c].contiguous().view(-1, output_dim),
                        images_discrete[:, c].contiguous().view(-1)
                    )
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

    return total_loss / total_batches



# Boucle d'entraînement principale
def main():
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    class_names = ['T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
                  'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']

    checkpoint_path = os.path.join(checkpoint_dir, 'pixelcnn_fashion_latest.pth')
    start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, device, output_dim)
        train_losses.append(train_loss)

        val_loss = evaluate(model, test_loader, criterion, device, output_dim)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch + 1, val_loss,
                       os.path.join(checkpoint_dir, 'pixelcnn_fashion_latest.pth'))

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss,
                           os.path.join(checkpoint_dir, 'pixelcnn_fashion_best.pth'))

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            selected_classes = list(range(n_classes))
            samples = generate_samples(model, selected_classes,
                                      shape=(in_channels, 28, 28), device=device)

            fig = visualize_samples(samples, [class_names[c] for c in selected_classes])
            fig.savefig(f'fashion_samples_epoch_{epoch+1}.png')
            plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('fashion_loss_history.png')
    plt.close()

    print("Entraînement terminé!")

    best_model_path = os.path.join(checkpoint_dir, 'pixelcnn_fashion_best.pth')
    load_checkpoint(model, optimizer, best_model_path)

    final_samples = generate_samples(model, range(n_classes),
                                    shape=(in_channels, 28, 28), device=device)

    fig = visualize_samples(final_samples, class_names)
    fig.savefig('fashion_final_samples.png')
    plt.close(fig)