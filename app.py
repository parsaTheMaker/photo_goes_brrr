import streamlit as st
from super_image import EdsrModel, ImageLoader
from PIL import Image
import torch
import io

# Title and Description
st.title("Image Enhancer (Super-Resolution)")
st.write("Enhance your images using the EDSR super-resolution model.")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model
    @st.cache_resource
    def load_model():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2)
        model = model.to(device)
        return model, device

    model, device = load_model()

    with st.spinner('Enhancing image...'):
        # Prepare the image for upscaling
        inputs = ImageLoader.load_image(image).to(device)

        # Perform upscaling
        with torch.no_grad():
            preds = model(inputs)

        # Convert preds to a PIL Image
        enhanced_image = ImageLoader.save_image(preds)

    # Display the enhanced image
    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

    # Download button for the enhanced image
    img_byte_arr = io.BytesIO()
    enhanced_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    st.download_button(
        label="Download Enhanced Image",
        data=img_byte_arr,
        file_name="enhanced_image.png",
        mime="image/png",
    )
