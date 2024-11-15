import streamlit as st
from super_image import MsrnModel, ImageLoader
from PIL import Image
import torch
import io

# Title and Description
st.title("Image Enhancer (4x Upscaler)")
st.write("Enhance your images using the MSRN model for 4x resolution scaling.")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the MSRN model
    @st.cache_resource
    def load_model():
        model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return model.to(device), device

    model, device = load_model()

    # Prepare the image for upscaling
    inputs = ImageLoader.load_image(image).to(device)

    # Perform upscaling
    with torch.no_grad():
        preds = model(inputs)

    # Save and display the enhanced image
    enhanced_image = ImageLoader.to_image(preds)
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
