import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time

from masked import MaskedConv2d
from condition import ConditionEncoder
from gated_pixelcnn import GatedPixelCNN
from sample import discretize_images
from checkpoint import load_checkpoint
from visualise_CIFAR import generate_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = 10
in_channels = 3
hidden_channels = 128
n_layers = 10
output_dim = 256

class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']

@st.cache_resource
def load_model():
    model = GatedPixelCNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_classes=n_classes,
        output_dim=output_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    import gzip
import shutil
import tempfile

@st.cache_resource
def load_model():
    model = GatedPixelCNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_classes=n_classes,
        output_dim=output_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    compressed_checkpoint = "./checkpoints/pixelcnn_best.pth.gz"

    if os.path.exists(compressed_checkpoint):
        with gzip.open(compressed_checkpoint, "rb") as f:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            shutil.copyfileobj(f, temp_file)
            temp_file.close()

            epoch, loss = load_checkpoint(model, optimizer, temp_file.name)
            os.remove(temp_file.name)  
            
        st.success(f"Modèle chargé avec succès (Epoch {epoch}, Loss {loss:.4f})")
        return model
    else:
        st.error("Aucun modèle pré-entraîné trouvé. Veuillez entraîner le modèle d'abord.")
        return None

def convert_sample_to_image(sample):
    img = sample.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img

def generate_and_display_sample(model, class_label, progress_placeholder=None):
    if progress_placeholder:
        progress_placeholder.text(f"Génération en cours pour la classe '{class_names[class_label]}'...")
    
    sample = generate_samples(model, [class_label], shape=(in_channels, 32, 32), device=device)[0]
    
    img = convert_sample_to_image(sample)
    if progress_placeholder:
        progress_placeholder.empty()
    
    return img

def main():
    st.set_page_config(
        page_title="Générateur PixelCNN CIFAR-10",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Générateur d'images PixelCNN pour CIFAR-10")
    
    st.markdown("""
    Cette application utilise un modèle Gated PixelCNN pour générer des images conditionnelles 
    basées sur les classes CIFAR-10.
    """)
    
    st.sidebar.title("Paramètres de génération")
    
    selected_class = st.sidebar.selectbox(
        "Choisissez une classe à générer:",
        range(len(class_names)),
        format_func=lambda x: class_names[x]
    )
    
    display_mode = st.sidebar.radio(
        "Mode d'affichage:",
        ["Unique", "Grille (plusieurs images)"]
    )
    
    num_samples = 1
    if display_mode == "Grille (plusieurs images)":
        num_samples = st.sidebar.slider("Nombre d'images à générer:", 2, 9, 4)
    
    model = load_model()
    
    if model is not None:
        if display_mode == "Unique":
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.subheader(f"Génération pour: {class_names[selected_class]}")
                
                generate_button = st.button("Générer une nouvelle image", use_container_width=True)
                
                if "current_image" not in st.session_state or generate_button:
                    progress_placeholder = st.empty()
                    st.session_state.current_image = generate_and_display_sample(
                        model, selected_class, progress_placeholder
                    )
                
                st.image(
                    st.session_state.current_image, 
                    caption=f"{class_names[selected_class]}",
                    use_column_width=True
                )
                
                pil_img = Image.fromarray((st.session_state.current_image * 255).astype(np.uint8))
                buf = pil_img.tobytes()
                st.download_button(
                    label="Télécharger cette image",
                    data=buf,
                    file_name=f"{class_names[selected_class]}_generated.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Détails du modèle")
                st.markdown("""
                Architecture: Gated PixelCNN
                
                Cette architecture utilise des convolutions masquées pour générer des images pixel par pixel,
                en respectant la dépendance causale (chaque pixel ne dépend que des pixels précédents).
                
                Comment ça marche:
                1. Le modèle prédit la distribution de probabilité pour chaque pixel
                2. Il échantillonne à partir de cette distribution
                3. Les prédictions suivantes tiennent compte des pixels déjà générés
                
                Conditionnement:
                Le modèle utilise les étiquettes de classe pour conditionner la génération,
                ce qui lui permet de créer des images spécifiques à chaque classe.
                """)
                
                # on affiche la distrib des pixels
                st.subheader("Distribution des pixels")
                fig, ax = plt.subplots(figsize=(8, 2))
                
                pixel_values = st.session_state.current_image.flatten()
                ax.hist(pixel_values, bins=50, alpha=0.7)
                ax.set_xlabel("Valeur de pixel (0-1)")
                ax.set_ylabel("Fréquence")
                st.pyplot(fig)
        
        else:
            st.subheader(f"Grille de {num_samples} images pour: {class_names[selected_class]}")
            
            if st.button("Générer des images", use_container_width=True):
                progress_placeholder = st.empty()
                

                cols = st.columns(min(3, num_samples))
                
                for i in range(num_samples):
                    col_idx = i % len(cols)
                    
                    with cols[col_idx]:
                        progress_placeholder.text(f"Génération de l'image {i+1}/{num_samples}...")
                        
                        img = generate_and_display_sample(model, selected_class)
                        
                        st.image(img, caption=f"{class_names[selected_class]} #{i+1}", use_column_width=True)
                        

                        pil_img = Image.fromarray((img * 255).astype(np.uint8))
                        buf = pil_img.tobytes()
                        st.download_button(
                            label="Télécharger",
                            data=buf,
                            file_name=f"{class_names[selected_class]}_{i+1}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                progress_placeholder.empty()
    
    st.sidebar.markdown("---")
    st.sidebar.header("À propos du modèle")
    st.sidebar.info("""
    **Gated PixelCNN**
    
    Ce modèle autorégressif génère des images en prédisant chaque pixel
    conditionné sur les pixels précédents et l'étiquette de classe.
    
    Entraîné sur le dataset CIFAR-10 contenant 60 000 images
    de taille 32x32 pixels réparties en 10 classes.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Informations techniques")
    st.sidebar.code(f"""
    Device: {device}
    Nb couches: {n_layers}
    Canaux cachés: {hidden_channels}
    Nb classes: {n_classes}
    Dim sortie: {output_dim}
    """)

if __name__ == "__main__":
    main()