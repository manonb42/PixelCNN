import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from masked import MaskedConv2d
from condition import ConditionEncoder
from gated_pixelcnn import GatedPixelCNN
from sample import generate_samples
from checkpoint import load_checkpoint
from visualize import visualize_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = 10
in_channels = 1
hidden_channels = 64
n_layers = 10
output_dim = 32


class_names = ['T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
              'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']


@st.cache_resource
def load_model():
    model = GatedPixelCNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_classes=n_classes,
        output_dim=output_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    return model, optimizer



def visualize_samples_for_streamlit(samples, class_labels=None):
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
    
    if n_samples == 1:
        axes = [axes]
        
    for i, sample in enumerate(samples):
        img = sample.squeeze(0).numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        if class_labels is not None and i < len(class_labels):
            axes[i].set_title(class_names[class_labels[i]])
    
    plt.tight_layout()
    return fig



st.title("Gated PixelCNN: Générateur d'images conditionnel")
st.markdown("""
Cette application utilise un modèle Gated PixelCNN conditionnel pour générer des images de vêtements 
en fonction de la classe choisie. Le modèle a été entraîné sur le dataset Fashion MNIST.
""")


st.sidebar.header("Options")


checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    st.warning(f"Le dossier {checkpoint_dir} n'existait pas et a été créé. Veuillez y placer vos fichiers de modèle.")


available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')] if os.path.exists(checkpoint_dir) else []

if not available_checkpoints:
    st.error("Aucun checkpoint trouvé dans le dossier ./checkpoints/")
    st.info("Veuillez placer au moins un fichier de modèle .pth dans le dossier checkpoints.")
else:
    selected_checkpoint = st.sidebar.selectbox(
        "Sélectionner un checkpoint",
        available_checkpoints
    )
    model, optimizer = load_model()
    checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
    epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
    
    st.sidebar.write(f"Epoch du checkpoint: {epoch}")
    st.sidebar.write(f"Perte: {loss:.4f}")
    st.sidebar.write(f"Dispositif: {device}")
    
    tab1, tab2 = st.tabs(["Générer des images", "Résultats d'entraînement"])
    
    with tab1:
        st.header("Générer de nouvelles images")
        st.subheader("Paramètres de génération")
        class_selection = st.multiselect(
            "Sélectionner les classes à générer",
            options=list(range(len(class_names))),
            format_func=lambda x: f"{x}: {class_names[x]}",
            default=[0]
        )
        
        n_samples_per_class = st.slider(
            "Nombre d'échantillons par classe",
            min_value=1,
            max_value=3,
            value=1
        )
        
        if st.button("Générer des images"):
            if class_selection:
                all_samples = []
                all_classes = []
                
                with st.spinner("Génération des images en cours..."):
                    for _ in range(n_samples_per_class):
                        for class_idx in class_selection:
                            condition = torch.tensor([class_idx], device=device)
                            
                            sample = model.sample(condition=condition, shape=(in_channels, 28, 28), device=device)
                            all_samples.append(sample.cpu().squeeze(0))
                            all_classes.append(class_idx)
                
                fig = visualize_samples_for_streamlit(all_samples, all_classes)
                st.pyplot(fig)
            else:
                st.warning("Veuillez sélectionner au moins une classe")
    
    with tab2:
        st.header("Résultats d'entraînement")
        
        loss_history_path = 'fashion_loss_history.png'
        if os.path.exists(loss_history_path):
            st.image(loss_history_path, caption="Historique des pertes d'entraînement")
        else:
            st.warning("Le fichier d'historique des pertes n'est pas disponible")
        
        sample_files = [f for f in os.listdir('.') if f.startswith('fashion_samples_epoch_') and f.endswith('.png')]
        
        if sample_files:
            st.subheader("Échantillons générés pendant l'entraînement")
            
            sample_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[-1].split('.')[0]))))
            
            selected_sample = st.selectbox(
                "Sélectionner une époque",
                sample_files,
                format_func=lambda x: f"Époque {x.split('_')[-1].split('.')[0]}"
            )
            
            st.image(selected_sample, caption=f"Échantillons de l'époque {selected_sample.split('_')[-1].split('.')[0]}")
        else:
            st.warning("Aucun fichier d'échantillons d'entraînement trouvé")
        
        final_samples_path = 'fashion_final_samples.png'
        if os.path.exists(final_samples_path):
            st.subheader("Échantillons finaux")
            st.image(final_samples_path, caption="Échantillons finaux pour chaque classe")
    

    with st.sidebar.expander("À propos de Gated PixelCNN"):
        st.write("""
        **Gated PixelCNN** est un modèle autorégressif qui génère des images pixel par pixel.
        
        La version conditionnelle permet de contrôler la génération en spécifiant une classe.
        
        Ce modèle utilise:
        - Des convolutions masquées pour respecter l'ordre de génération autorégressif
        - Des connexions "gated" qui améliorent la capacité du modèle
        - Un encodeur de condition pour intégrer l'information de classe
        """)