
import torch
import streamlit as st
from bitcanvas import bitcanvas
import os

from ..pixelcnn import Model, generate_image

# Cette ligne corrige un bug ésotérique
# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def models(digit):
    return torch.load(f"models/only_{digit}.pth", map_location=device)


st.title("PixelCNN")

# Slider to select the epoch [0-200]
digit = st.slider("Digit", 0, 9, 0)
epoch = st.slider("Epoch", 1, len(models(digit)), 1)
variance = st.slider("Variance", 1, 64, 1)

model = Model().to(device)

model.load_state_dict(models(digit)[epoch])


col1, col2 = st.columns(2)
with col1:
    data = bitcanvas(28, 28, key="foo")
    image = data["image"]
    marker = data["marker"]

with col2:
    if image:
        image = [[[sum(val)/3/255 for val in row] for row in image]]
        image = torch.tensor(image).to(device)
        image = image.unsqueeze(0)
        skip = marker['y']*28 + marker['x']
        image = generate_image(model, device, skip=skip, image=image, k=2**(64-variance))
        image = [[[float(val)*255]*3 for val in row] for row in image]
        bitcanvas(28, 28, image=image, key="baz")
    else:
        st.markdown("Waiting for image to load...")
