import streamlit as st
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

# Cache the model to load it only once
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32  # Adjust dtype for CPU/GPU
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)

# Load the upscaler model
pipe = load_model()

# Streamlit app
st.title("Image Upscaling with Stable Diffusion x4")
st.write("Upload an image and upscale it using the Stable Diffusion x4 Upscaler.")

# Upload image
uploaded_file = st.file_uploader("Choose an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Original Image", use_column_width=True)

    # Perform upscaling
    with st.spinner("Upscaling the image..."):
        upscaled_image = pipe(prompt="", image=input_image).images[0]

    # Display the upscaled image
    st.image(upscaled_image, caption="Upscaled Image", use_column_width=True)

    # Download button for the upscaled image
    st.download_button(
        label="Download Upscaled Image",
        data=upscaled_image.tobytes(),
        file_name="upscaled_image.png",
        mime="image/png"
    )
