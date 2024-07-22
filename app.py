import os
import numpy as np
# streamlit app
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
# image display
from PIL import Image, ImageDraw
# image inpainting models
from test_UNet import get_result_UNet
from test_GAN import get_result_GAN
from test_DIP import get_result_DIP

IMAGE_SIZE = 96
CROP = 96
DISPLAY_FACTOR = 3
MASK_RATIO = 0.2
UNET_PATH = './models/unet.pth'
UNET_GPATH = os.environ['UNET_GPATH']
GAN_G_PATH = './models/generator_alpha1.0.pth'
GAN_G_GPARH = os.environ['GAN_G_GPARH']


@st.cache_data
def load_UNet():
    """
    Download U-Net model from Google drive.
    """
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(UNET_PATH):
        with st.spinner("Downloading model... This may take a while! Don't stop it!"):
            import gdown
            gdown.download(UNET_GPATH, UNET_PATH, quiet=True)

@st.cache_data
def load_GAN():
    """
    Download GAN generator from Google drive. The discriminator is also on Google drive,
    but it is not used hence not downloaded.
    """
    os.makedirs('models', exist_ok=True)
    if not os.path.exists(GAN_G_PATH):
        with st.spinner("Downloading model... This may take a while! Don't stop it!"):
            import gdown
            gdown.download(GAN_G_GPARH, GAN_G_PATH, quiet=True)

def get_resize_size(dim):
    """
    Get the width and height of image to be displayed. 
    Can be used to resize original image by display factor.

    :param dim: Tuple of two integers, width and height of original image.
    :return: Tuple of two integers, width and height of image to be displayed.
    """
    width, height = dim
    if width < height:
        return IMAGE_SIZE, int(IMAGE_SIZE * height / width)
    else:
        return int(IMAGE_SIZE * width / height), IMAGE_SIZE

def get_crop_coords(dim):
    """
    Get the coordinates of the top left and bottom right corners for centre cropping. 
    Can be used to crop a square out of an image.

    :param dim: Tuple of two integers, width and height of original image.
    :return: Tuple of four integers, (x1, y1, x2, y2).
    """
    width, height = dim
    x1 = (width - CROP) // 2
    y1 = (height - CROP) // 2
    return x1, y1, x1 + CROP, y1 + CROP

def get_rect_coords(point, dim):
    """
    Get the coordinates of the top left and bottom right corners for masking.
    Can be used to display a square mask within an image according to a mask ratio.

    :param point: Tuple of two integers, 
        (x, y) coordinates that coorespond to the centre of the mask.
    :param dim: Tuple of two integers, width and height of image to be displayed.
    :return: Tuple of four integers, (x1, y1, x2, y2).
    """
    x, y = point
    width, height = dim
    MASK_SIZE = int(np.sqrt((IMAGE_SIZE * DISPLAY_FACTOR)**2 * MASK_RATIO))
    MASK_HALF = MASK_SIZE // 2
    x_c = min(max(x, MASK_HALF), width - MASK_HALF)
    y_c = min(max(y, MASK_HALF), height - MASK_HALF)
    return x_c - MASK_HALF, y_c - MASK_HALF, x_c + MASK_HALF, y_c + MASK_HALF

def get_mask_coords(point, dim):
    """
    Get the coordinates of the top left and bottom right corners for masking.
    Can be used to mask a square out of an image according to a mask ratio.

    :param point: Tuple of two integers, 
        (x, y) coordinates that coorespond to the centre of the mask.
    :param dim: Tuple of two integers, width and height of original image.
    :return: Tuple of four integers, (x1, y1, x2, y2).
    """
    x, y = point
    width, height = dim
    x = x // DISPLAY_FACTOR
    y = y // DISPLAY_FACTOR
    MASK_SIZE = int(np.sqrt(IMAGE_SIZE**2 * MASK_RATIO))
    MASK_HALF = MASK_SIZE // 2
    x_c = min(max(x, MASK_HALF), width - MASK_HALF)
    y_c = min(max(y, MASK_HALF), height - MASK_HALF)
    return x_c - MASK_HALF, y_c - MASK_HALF, x_c + MASK_HALF, y_c + MASK_HALF


# download models
load_UNet()
load_GAN()
# display
st.title("Image Inpainting")
st.markdown("### Step 1: Upload an image")
file = st.file_uploader("Upload your own image", type=['png', 'jpg', 'jpeg'])
if file is not None:
    # load image
    image = Image.open(file)
    # crop image to fit model input size
    resize_size = get_resize_size(image.size)
    image = image.resize(resize_size)
    crop_coords = get_crop_coords(image.size)
    image = image.crop(crop_coords)
    image = image.convert('RGB')
    display_size = (IMAGE_SIZE * DISPLAY_FACTOR, IMAGE_SIZE * DISPLAY_FACTOR)
    image_display = image.resize(display_size)
    st.image(image_display)
    # display image
    st.markdown("### Step 2: Create a mask")
    st.write("Select the position of the mask by clicking on the image. The mask is limited to a square that covers 20\% of the image.")
    draw = ImageDraw.Draw(image_display)
    # display mask
    if 'point' in st.session_state:
        point = st.session_state["point"]
        rect_coords = get_rect_coords(point, image_display.size)
        draw.rectangle(rect_coords, fill="black")
    # get coordinates from user
    value = streamlit_image_coordinates(image_display, key="pil")
    if value:
        # save coordinates
        st.markdown("### Step 3: Auto-generate")
        st.write("Select model and click on the 'Generate' button to see the result.")
        point = value["x"], value["y"]
        # select image inpainting model
        model = st.radio(
            "Image inpainting model", 
            ["Autoencoder", "GAN", "DIP"],
            captions=[
                "Autoencoder with U-Net architecture and MAE loss",
                "Generative Adversarial Network using U-Net as the generator and Resnet18 as the discriminator",
                "Deep Image Prior with U-Net architecture and MAE loss"
            ]
        )
        if model == 'DIP':
            epochs = st.number_input(
                label="Number of epochs", min_value=1, max_value=2000, value=800, step=100
            )
        if 'point' not in st.session_state or point != st.session_state["point"]:
            # save coordinates and update mask display
            st.session_state["point"] = point
            st.rerun()
        if st.button("Generate", type="primary"):
            # get mask coordinates on original image
            x0, y0, x1, y1 =  get_mask_coords(point, image.size)
            mask = np.ones(image.size)
            mask[y0:y1, x0:x1] = 0
            # generate and diaplay result
            with st.spinner("Running... This may take a while!"):
                if model == 'DIP':
                    pred, error = get_result_DIP((image, mask), epochs=epochs)
                elif model == 'GAN':
                    pred, error = get_result_GAN(GAN_G_PATH, (image, mask))
                else:
                    pred, error = get_result_UNet(UNET_PATH, (image, mask))
            st.image(pred.resize(display_size))
            # display error
            st.metric("Mean Absolute Error (MAE)", f"{error['mae']:.3f}")
            st.metric("Structural Similarity Index (SSIM)", f"{error['ssim']:.3f}")
