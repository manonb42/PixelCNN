import torch
from torch.distributions.beta import Beta
import torch.nn.functional as F

def empty_image(shape, device):
    """
    Créé une image vide (entièrement noire)
    :param shape: Les dimensions de l'image à créér `(nb_channels, height, width)`
    par example `(1,28,28)` pour MNIST et `(3,32,32)` pour CIFAR
    """
    return torch.zeros(1, *shape).to(device)



def generate_image_bin(model, image, skip=0, var=50.0):
    """ Utilise le modèle pour générer une image (ou sa suite) en teinte de gris

    :param model: le modèle à utiliser
    :param image: l'image à compléter. voir `empty_image` pour obtenir une image vide.
    :param skip: le nombre de pixel à considérer comme pré-rempli
    :param var: la variance de la loi utilisée
    """
    model.eval()

    with torch.no_grad():
        for i in range(skip, image.size().numel()):
            x, y = divmod(i, image.size(-1))
            out = model(image)
            prob = torch.sigmoid(out[0, 0, x, y])
            image[0, 0, x, y] = Beta(prob*var, (1-prob)*var).sample() if var != 0 else prob

    return image.cpu().numpy()[0, 0]

def generate_image_multi(model, image, condition=None):
    """ Utilise le modèle pour générer une image (ou sa suite) en teinte de gris

    :param model: le modèle à utiliser
    :param image: l'image à compléter. voir `empty_image` pour obtenir une image vide.
    :param skip: le nombre de pixel à considérer comme pré-rempli
    :param var: la variance de la loi utilisée
    """

    model.eval()

    with torch.no_grad():
        for h in range(image.size(1)):
            for w in range(image.size(2)):
                for c in range(image.size(0)):
                    out = model(image, condition)
                    probs = F.softmax(out[0, c, h, w], dim=0)
                    image[0, c, h, w] = torch.multinomial(probs, 1).float() / out.size(-1)
