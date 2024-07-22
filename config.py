import numpy as np
import torch
from torchvision import transforms

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = "datasets/JPEGImages/"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRAIN_SIZE = 0.8
IMG_SIZE = 96
MASK_RATIO = 0.2
BATCH_SIZE = 32
RANDOM_STATE = 484

# data augmentation
UNNORM = transforms.Normalize(
    mean=-np.divide(MEAN, STD),
    std=np.divide(1, STD),
    inplace=True
)
IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(144),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, hue=0.1),
    transforms.Normalize(MEAN, STD)
])
MASK_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(144),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomErasing(
        p=1.0, scale=(MASK_RATIO, MASK_RATIO), ratio=(1.0, 1.0)
    ),
])
TEST_IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
TEST_MASK_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])
