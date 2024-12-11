import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, labels, img_dir, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = torch.FloatTensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label
