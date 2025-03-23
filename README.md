# PixelCNN

## Description
PixelCNN est un modèle génératif autorégressif qui modélise la distribution de probabilité jointe des pixels dans une image en la décomposant en un produit de distributions conditionnelles. Le modèle génère les images pixel par pixel, dans un ordre séquentiel (généralement de gauche à droite et de haut en bas), où chaque nouveau pixel est conditionné par tous les pixels précédemment générés.

Contrairement aux GAN qui utilisent un réseau discriminateur ou aux VAE qui compressent les données dans un espace latent, PixelCNN optimise directement la vraisemblance exacte des données. Bien que le processus de génération soit séquentiel et donc relativement lent, l'entraînement peut être parallélisé puisque les vraies valeurs de tous les pixels sont connues pendant cette phase.

## Fonctionnalités
Ce dépôt contient des implémentations de modèles PixelCNN pour la génération d'images pixel par pixel sur différents jeux de données avec les caractéristiques suivantes :
- Génération d'images complètes
- **Complétion interactive**: possibilité pour l'utilisateur de dessiner le début d'un chiffre et le modèle PixelCNN effectuera automatiquement la complétion du dessin

## Modèles implémentés
Ce projet comprend deux implémentations principales :
1. **PixelCNN standard** - Utilisé pour la génération d'images sur MNIST
2. **Gated PixelCNN** - Une version améliorée utilisant des mécanismes de porte (gating) pour de meilleures performances, implémentée pour Fashion-MNIST et CIFAR-10

## Jeux de données
Les modèles ont été entraînés sur les jeux de données suivants :
- **MNIST** - Chiffres manuscrits (PixelCNN standard)
- **Fashion-MNIST** - Articles vestimentaires (Gated PixelCNN)
- **CIFAR-10** - Images en couleur de 10 classes (Gated PixelCNN)

## Installation et utilisation
Pour exécuter l'application Streamlit:
```sh
$ uv sync 
$ uv run streamlit run start_streamlit.py
```

En cas de problème avec `bitcanvas`:
```sh
$ cd src/app/canvas
$ uv pip install -e .
```

## Références
- van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel Recurrent Neural Networks. ArXiv:1601.06759 [Cs].
- van den Oord, A., Kalchbrenner, N., Espeholt, L., Vinyals, O., Graves, A., & Kavukcuoglu, K. (2016). Conditional Image Generation with PixelCNN Decoders. ArXiv:1606.05328 [Cs].
