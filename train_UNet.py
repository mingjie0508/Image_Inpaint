import os
from sklearn.model_selection import train_test_split
from InpaintDataset import InpaintDataset
from UNet import UNet
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss as mae
from config import *


def train_unet(model, dataloader, optimizer, epochs, checkpoint_path):
    """
    Training loop for UNet. Performs training and saves final model.

    :param model: Pytorch model.
    :param dataloader: Training dataloader that loads images and masks.
    :param optimizer: Pytorch optimizer.
    :param epochs: Number of epochs.
    :param checkpoint_path: Output checkpoint path.
    :return: List of training losses.
    """
    # losses for each epoch
    losses = []
    for epoch in range(epochs):
        print('-'*10)
        print(f'Epoch {epoch + 1}/{epochs}:')
        model.train()
        running_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            masked_x = x * y
            output = model(masked_x)
            loss = model.get_loss(output, x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f'Train loss:', round(epoch_loss, 4))
        
    # save final model
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'loss': epoch_loss},
        checkpoint_path
    )
    return losses


def test_unet(model, dataloader):
    """
    Test loop for UNet. Prints test loss and evaluation metrics.
    Returns predicted images.
    
    :param model: Pytorch model.
    :param dataloader: Test dataloader that loads images and masks.
    :return: List of Tensors, predicted images.
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    pred_list = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            masked_x = x * y
            output = model(masked_x)
            loss = model.get_loss(output, x, y)

            running_loss += loss.item()
            running_mae += mae(masked_x + (1-y)*output, x).item()
            pred_list.append(output.cpu())

    pred_list = torch.cat(pred_list)
    loss = running_loss / len(dataloader)
    mae_score = running_mae / len(dataloader)
    print('Test loss:', round(loss, 4))
    print('Test MAE:', round(mae_score, 4))
    return pred_list


if __name__ == "__main__":
    # get the list of all image paths
    paths = [os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH)]
    # split paths into training and testing
    paths_train, paths_test = train_test_split(
        paths, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    # configure datasets
    train_dataset = InpaintDataset(paths_train, IMG_TRANSFORM, MASK_TRANSFORM)
    test_dataset = InpaintDataset(paths_test, IMG_TRANSFORM, MASK_TRANSFORM)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # configure model
    unet = UNet(in_channels=3, out_channels=3)
    unet = unet.to(DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4, weight_decay=0.0)

    unet_path = "./models/unet.pth"
    checkpoint = torch.load(unet_path)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # train UNet, print training loss
    epochs = 16  # can modify
    print("Training UNet")
    unet_train_loss = train_unet(
        unet, train_dataloader, optimizer, epochs=epochs,
        checkpoint_path=unet_path
    )

    # test UNet
    print("Testing UNet")
    unet_pred = test_unet(unet, test_dataloader)
