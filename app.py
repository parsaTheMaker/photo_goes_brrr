import streamlit as st
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

# Load the Stable Diffusion x4 Upscaler model
model_id = "stabilityai/stable-diffusion-x4-upscaler"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

st.title("Image Upscaling with Stable Diffusion x4 Upscaler")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Original Image", use_column_width=True)

    # Upscale the image
    with st.spinner("Upscaling..."):
        upscaled_image = pipe(prompt="", image=input_image).images[0]
    st.image(upscaled_image, caption="Upscaled Image", use_column_width=True)
