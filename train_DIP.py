import os
from sklearn.model_selection import train_test_split
from InpaintDataset import InpaintDataset
from DIP import DIP
from torch.utils.data import DataLoader
from torch.nn.functional import l1_loss as mae
from config import *


def train_dip(model, test_sample, optimizer, epochs, checkpoint_path):
    """
    Training loop for DIP. Performs training for a single image and 
    saves final model.

    :param model: Pytorch model.
    :param test_sample: Tuple of DIP input, target, and mask.
    :param optimizer: Pytorch optimizer.
    :param epochs: Number of epochs.
    :param checkpoint_path: Output checkpoint path.
    :return: List of losses.
    """
    # loss for each epoch
    losses = []
    x, target, mask = test_sample
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

def test_dip(model, test_sample):
    """
    Test loop for DIP. Prints test loss and evaluation metrics.
    Returns predicted image.

    :param model: Pytorch model.
    :param test_sample: Tuple of DIP input, target, and mask.
    :return: Tensor, predicted image.
    """
    x, target, mask = test_sample
    x, target, mask = x.to(DEVICE), target.to(DEVICE).float(), mask.to(DEVICE).float()
    model.eval()
    with torch.no_grad():
        output = model(x)

        loss = model.get_loss(output, target, mask).item()
        mae_score = mae(mask*target + (1-mask)*output, target).item()
    
    print('Test loss:', round(loss, 4))
    print('Test MAE:', round(mae_score, 4))
    return output[0]


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

    # run DIP
    n_plot = 3
    epochs = 700  # can modify
    torch.manual_seed(1)
    imgs, masks = next(iter(test_dataloader))
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
    pred = []
    # for each test image, run deep image prior
    print(f"DIP, n_iter = {epochs}")
    for i in range(n_plot):
        print("-"*10)
        print("Sample", i+1)
        dip = DIP(in_channels=3, out_channels=3)
        dip = dip.to(DEVICE)
        optimizer = torch.optim.Adam(dip.parameters(), lr=1e-4, weight_decay=0.0)
        dip_path = f"./models/dip{i}.pth"

        # current image and mask
        img, mask = imgs[[i]], masks[[i]]
        # input is sampled from Unif (0, 0.1)
        x = torch.rand(*img.shape) * 0.1
        # predict
        dip_loss = train_dip(
            dip, (x, img, mask), optimizer, epochs=epochs, checkpoint_path=dip_path
        )
        with torch.no_grad():
            pred.append(test_dip(dip, (x, img, mask)))
