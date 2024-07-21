from UNet import UNet
from sklearn.metrics import mean_absolute_error as mae
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as tF
from config import *


def predict_UNet(model, data):
    """
    Perform inpainting on a single image given an image and a mask.

    :param model: PyTorch model, U-Net.
    :param data: Tuple, an image and a mask.
    :return: Tensor, a single predicted image.
    """
    model.eval()
    with torch.no_grad():
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        masked_x = x * y
        output = model(masked_x)
    # loss = model.get_loss(output, x, y)
    # error = mae(masked_x + (1-y)*output, x).item()

    return output[0].cpu()


def get_result_UNet(model_path, data):
    """
    Perform inpainting on a single image given an image and a mask.

    :param model_path: str, path to model checkpoint.
    :param data: Tuple, an image and a mask.
    :return: Tuple, predicted Pillow image, and dict of metrics, 
        including 'mae' and 'ssim'.
    """
    # prepare data
    image, mask = data
    image = TEST_IMG_TRANSFORM(image).float().unsqueeze(0)
    mask = TEST_MASK_TRANSFORM(mask).float().unsqueeze(0)
    
    # configure model
    unet = UNet(in_channels=3, out_channels=3)
    unet = unet.to(DEVICE)
    checkpoint = torch.load(model_path)
    unet.load_state_dict(checkpoint['model_state_dict'])

    # get model output
    output = predict_UNet(unet, (image, mask))
    image = UNNORM(image[0])
    mask = mask[0]
    masked_img = image * mask
    output = UNNORM(output)
    pred = masked_img + (1 - mask) * output

    # calculate error between original image and predicted image
    mae_error = mae(image.flatten(), pred.flatten())
    ssim_error = ssim(image.numpy(), pred.numpy(), channel_axis=0)
    error = {'mae': mae_error, 'ssim': ssim_error}
    return tF.to_pil_image(pred), error


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # test image
    img_path = 'C:/Users/22156/Desktop/image1.jpg'
    img = Image.open(img_path)
    # test mask
    mask = np.ones(img.size)
    mask[40:84,3:47] = 0

    unet_path = './models/unet.pth'
    pred, error = get_result_UNet(unet_path, (img, mask))
    print(f"MAE: {error['mae']:.3f}")
    print(f"SSIM: {error['ssim']:.3f}")
    pred.show()
