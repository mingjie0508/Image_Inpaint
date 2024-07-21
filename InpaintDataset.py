import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# dataset class
class InpaintDataset(Dataset):
    def __init__(self, paths, img_transform=None, mask_transform=None):
        self.paths = paths
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        mask = np.ones(img.size)
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask
    

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    import torchvision.transforms.functional as tF
    from config import *

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

    # sample
    imgs, masks = next(iter(train_dataloader))
    img1, mask1 = UNNORM(imgs[0]), masks[0]
    img2, mask2 = UNNORM(imgs[1]), masks[1]

    # display
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1)
    plt.title('sample1 - masked image')
    ax1.imshow(tF.to_pil_image(img1 * mask1))
    ax2 = fig.add_subplot(2,2,2)
    plt.title('sample1 - target')
    ax2.imshow(tF.to_pil_image(img1))
    ax3 = fig.add_subplot(2,2,3)
    plt.title('sample2 - masked image')
    ax3.imshow(tF.to_pil_image(img2 * mask2))
    ax4 = fig.add_subplot(2,2,4)
    plt.title('sample2 - target')
    ax4.imshow(tF.to_pil_image(img2))
    plt.show()
