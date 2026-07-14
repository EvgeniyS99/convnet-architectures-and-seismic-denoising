import torch
import torch.nn as nn
import torchvision
import cv2
from PIL import Image
from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

def find_train_val_files():
    """
    Finds and adds train and val images, labels to list
    """
    files = []
    labels = []
    for root, dirs, file in os.walk('imagenette2-320/train'):
        for subfile in file:
            if 'ILSVR' not in subfile:
                files.append(os.path.join(root, subfile))
                labels.append(root[-9::])
    
    return files, labels

def find_test_files():
    """
    Finds and adds test images, labels to list
    """
    files = []
    labels = []
    for root, dirs, file in os.walk('imagenette2-320/val'):
        for subfile in file:
            if 'ILSVR' not in subfile:
                files.append(os.path.join(root, subfile))
                labels.append(root[-9::])
    
    return files, labels

class ImageNet(Dataset):
    def __init__(self, files, labels, mode, size=224):
        self.files = files
        self.labels = labels
        self.mode = mode
        self.label_encoder = LabelEncoder() 
        self.label_encoder.fit(self.labels)
        self.size = size
    
    def __len__(self):
        return len(self.files)
    
    def load_sample(self, file):
        image = Image.open(file).convert('RGB')
        image.load()
        return image

    def __getitem__(self, idx):
        
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        ])
        
        x = self.load_sample(self.files[idx])
        x = transform(x)
        label = self.labels[idx]
        label_id = self.label_encoder.transform([label]).item()
    
        return x, label_id
    
    def _prepare_sample(self, image):
        
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)
    
class AdeDataset(Dataset):
    
    def __init__(self, mode, root='ADEChallengeData2016', transforms=None, size=256):
        self.images_path = os.path.join(root, f'images/{mode}')
        self.masks_path = os.path.join(root, f'annotations/{mode}')
        self.images = sorted(os.listdir(self.images_path))
        self.masks = sorted(os.listdir(self.masks_path))
        self.mode = mode
        self.size = size
        self.transforms = transforms
        self.basic_transforms = A.Compose([
            A.Resize(size, size),
            #A.Normalize(mean=[], std=[]),
            ToTensorV2()
        ])
        
        for file in self.images:
            if file.endswith('.ipynb_checkpoints'):
                self.images.remove(file)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = cv2.imread(os.path.join(self.images_path, self.images[idx]), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_path, self.masks[idx]), cv2.IMREAD_UNCHANGED)
        
        if self.transforms is not None and self.mode == 'training':
            t = self.transforms(image=image, mask=mask)
            image = t['image'].float()
            mask = t['mask'].long()
        elif self.transforms is not None and self.mode == 'validation':
            t = self.basic_transforms(image=image, mask=mask)
            image = t['image'].float()
            mask = t['mask'].long()
        else:
            t = self.basic_transforms(image=image, mask=mask)
            image = t['image'].float()
            mask = t['mask'].long()
        
        return image, mask
    
def basic_transforms(size):
    
    transforms = A.Compose([
            A.Resize(size, size),
            #A.Normalize(mean=[], std=[]),
            ToTensorV2()
    ])
    
    return transforms

def augmentations_transforms(size):
    
    transforms = A.Compose([
        A.Resize(size, size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])
    
    return transforms