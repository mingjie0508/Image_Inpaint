from DIP import DIP
from sklearn.metrics import mean_absolute_error as mae
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as tF
from config import *


def train_dip(model, data, optimizer, epochs):
    """
    Training loop for DIP. Perform training on a single image.

    :param model: PyTorch model, U-Net.
    :param data: Tuple, random noise, an image, and a mask.
    :return: List of floats, training losses for each epoch.
    """
    # loss for each epoch
    losses = []
    x, target, mask = data
    x, target, mask = x.to(DEVICE), target.to(DEVICE).float(), mask.to(DEVICE).float()
    for epoch in range(epochs):
        # print('-'*10)
        # print(f'Epoch {epoch + 1}/{epochs}:')
        model.train()
        output = model(x)
        loss = model.get_loss(output, target, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        losses.append(epoch_loss)        
    # print(f'Train loss:', round(epoch_loss, 4))
    return losses


def predict_DIP(model, data):
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
    return output[0].cpu()


def get_result_DIP(data, epochs=800):
    """
    Perform inpainting on a single image given an image and a mask.

    :param data: Tuple, an image and a mask.
    :param epochs: int, number of epochs.
    :return: Tuple, predicted Pillow image, and dict of metrics, 
        including 'mae' and 'ssim'.
    """
    # prepare data
    image, mask = data
    image = TEST_IMG_TRANSFORM(image).float().unsqueeze(0)
    mask = TEST_MASK_TRANSFORM(mask).float().unsqueeze(0)
    x = torch.rand(*image.shape) * 0.1
    
    # configure model
    dip = DIP(in_channels=3, out_channels=3)
    dip = dip.to(DEVICE)
    optimizer = torch.optim.Adam(dip.parameters(), lr=1e-3, weight_decay=0.0)

    # get model output
    dip_loss = train_dip(
        dip, (x, image, mask), optimizer, epochs=epochs
    )
    output = predict_DIP(dip, (image, mask))
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

    EPOCHS = 800
    pred, error = get_result_DIP((img, mask), EPOCHS)
    print(f"MAE: {error['mae']:.3f}")
    print(f"SSIM: {error['ssim']:.3f}")
    pred.show()
