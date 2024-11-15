import streamlit as st
from PIL import Image
import torch
from realesrgan import RealESRGAN
import io

# Title and Description
st.title("Fast Image Upscaler")
st.write("Enhance your images with a lightweight pre-trained Real-ESRGAN model.")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RealESRGAN(device, scale=2)  # Use x2 upscaling
        model.load_weights("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x2plus.pth")
        return model

    model = load_model()

    with st.spinner("Upscaling image..."):
        # Perform enhancement
        enhanced_image = model.predict(image)

    # Display the enhanced image
    st.image(enhanced_image, caption="Upscaled Image", use_column_width=True)

    # Prepare the image for download
    img_byte_arr = io.BytesIO()
    enhanced_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Download button for the enhanced image
    st.download_button(
        label="Download Upscaled Image",
        data=img_byte_arr,
        file_name="upscaled_image.png",
        mime="image/png",
    )
