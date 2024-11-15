import streamlit as st
from super_image import MsrnModel, EdsrModel, ImageLoader
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
import io
import logging

# Suppress PyTorch and Hugging Face warnings
logging.getLogger('torch').setLevel(logging.ERROR)

# Title and Description
st.title("Image Enhancer (Super-Resolution)")
st.write("Enhance your images using various super-resolution models.")

# Sidebar for Model Selection
st.sidebar.title("Model Selection")
model_options = {
    'MSRN x2': ('MSRN', 'eugenesiow/msrn', 2),
    'MSRN x4': ('MSRN', 'eugenesiow/msrn', 4),
    'EDSR x2': ('EDSR', 'eugenesiow/edsr', 2),
    'EDSR x4': ('EDSR', 'eugenesiow/edsr', 4),
    # Add more models and scales as needed
}
selected_model_option = st.sidebar.selectbox('Select Super-Resolution Model', list(model_options.keys()))
model_type, model_name, scale = model_options[selected_model_option]

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load the model
    @st.cache_resource
    def load_model(model_type, model_name, scale):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'MSRN':
            model = MsrnModel.from_pretrained(model_name, scale=scale).to(device)
        elif model_type == 'EDSR':
            model = EdsrModel.from_pretrained(model_name, scale=scale).to(device)
        else:
            st.error(f"Unsupported model type: {model_type}")
            return None, device
        return model, device

    model, device = load_model(model_type, model_name, scale)

    if model is not None:
        with st.spinner('Enhancing image...'):
            # Prepare the image for upscaling
            inputs = ImageLoader.load_image(image)
            if device.type == 'cuda':
                inputs = inputs.to(device)

            # Perform upscaling
            with torch.no_grad():
                preds = model(inputs)

            # Process the output
            if preds.dim() == 4:
                preds = preds.squeeze(0)
            if preds.device != torch.device('cpu'):
                preds = preds.cpu()

            # Convert preds to a PIL Image
            enhanced_image = to_pil_image(preds)

            # Clean up to free memory
            del inputs, preds
            torch.cuda.empty_cache()

        # Display the enhanced image
        st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)

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
