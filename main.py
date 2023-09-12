from dataset import CUB
import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as utils


IMAGE_SIZE = 224
# Calculated by using cal.py
TRAIN_MEAN = [ 0.43237713785804116, 0.49941626449353244, 0.48560741861744905]
TRAIN_STD = [0.2665100547329813, 0.22770540015765814, 0.2321024260764962]
TEST_MEAN = [0.4311430419332438, 0.4998156522834164, 0.4862169586881995]
TEST_STD = [0.26667253517177186, 0.22781080253662814, 0.23264268069040475]

path = '/root/SharedData/datasets/CUB_200_2011'

train_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

test_transforms = transforms.Compose([
    transforms.ToCVImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(TEST_MEAN,TEST_STD)
    ])

train_dataset = CUB(
        path,
        train=True,
        transform=train_transforms,
        target_transform=None
    )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=8,
    shuffle=True
)

test_dataset = CUB(
        path,
        train=False,
        transform=test_transforms,
        target_transform=None
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=8,
    shuffle=True
)



#Save some pic to check
batch_iterator = iter(train_dataloader)
images, _ = next(batch_iterator)  
num_images_to_save = 4
images_to_save = images[:num_images_to_save]

def denormalize(images, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return images * std + mean


denormalized_images = denormalize(images_to_save, TRAIN_MEAN, TRAIN_STD).numpy()
for i in range(num_images_to_save):
    image = denormalized_images[i]
    image = (image * 255).astype(np.uint8).transpose(1, 2, 0) 
    image = Image.fromarray(image)  
    image.save(f"image_{i+1}.jpg")  

print(f"{num_images_to_save} images saved.")
