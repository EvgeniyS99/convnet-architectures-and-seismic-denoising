import torch
import torch.nn as nn
import torchvision
from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torchvision.io import read_image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.patches as patches
from tqdm import tqdm, tqdm_notebook
import pickle

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
    def __init__(self, files, labels, mode):
        self.files = files
        self.labels = labels
        self.mode = mode
        self.label_encoder = LabelEncoder() 
        self.label_encoder.fit(self.labels)
            
    def __len__(self):
        return len(self.files)
    
    def load_sample(self, file):
        image = Image.open(file).convert('RGB')
        image.load()
        return image

    def __getitem__(self, idx):
        #mg_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
