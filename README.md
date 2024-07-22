# Image_Inpaint
Image inpainting with Autoencoder / GAN / DIP.\
Image inpainting app: [https://imageinpaint-484.streamlit.app](https://imageinpaint-484.streamlit.app/)

The goal of inpainting is to complete missing or damaged parts of an image. For simplicity, the damaged parts in this project are assumed to be a square that covers $20\%$ of an image.

This project takes on three image inpainting techniques:
- [U-Net](https://arxiv.org/abs/1505.04597) with MAE loss
- GAN + U-Net (a discriminator appended to U-Net)
- [Deep image prior](https://arxiv.org/abs/1711.10925) with MAE loss

### Installation
Training requires GPU.

Install libraries
```
pip install -r requirements.txt
```

### Dataset
RGB images from the VOC2012 dataset. Only a small part of the dataset is uploaded here.

Directory: [datasets/JPEGImages](https://github.com/mingjie0508/Image_Inpaint/tree/main/datasets/JPEGImages)

### Run
Start training
```
# modify configuration in config.py
# modify model paths in train_UNet.py/train_GAN.py/train_DIP.py

python train_UNet.py
python train_GAN.py
python train_DIP.py
```

Testing does not require GPU. You may visit the app [here](https://imageinpaint-484.streamlit.app/), which does not require installation, or run the app locally
```
# modify model path in app.py
# run app

streamlit run app.py
```
