import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from CNN import create_model
from dataset import ChestXRayDataset
from PIL import Image
import numpy as np
import random

# Set random seeds for reproducibility
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Paths
DATA_CSV = "Data/Data_Entry_2017_v2020.csv"
IMAGES_DIR = "Data/images/"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
# if this takes too long reduce image size to 512 or 224
IMAGE_SIZE = 512

# our dataset has 14 classes corresponding to the 14 diseases outlined in our write up
NUM_CLASSES = 14  

# Pytorch device configuration for mac devices mps will use hardware acceleration
device = torch.device('mps')
print(f'Using device: {device}')

# First we will load the data to pandas dataframe 
images = pd.read_csv(DATA_CSV)
print(images.head())

print("Sample image name:", images.iloc[0]['Image Index'])
print("Path exists:", os.path.exists(os.path.join(IMAGES_DIR, images.iloc[0]['Image Index'])))



images.columns = [
    "Image Index", "Finding Labels", "Follow-up #", "Patient ID",
    "Patient Age", "Patient Gender", "View Position",
    "OriginalImageWidth", "OriginalImageHeight",
    "OriginalImagePixelSpacing_x", "OriginalImagePixelSpacing_y"
]



# we are only taking a subset of the data for faster training

mlb = MultiLabelBinarizer(classes=[
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
])

# split the labels and encode them from the data
images['Finding Labels'] = images['Finding Labels'].str.split('|')
y = mlb.fit_transform(images['Finding Labels'])

# debug print statements
print("Classes:", mlb.classes_)
print("Shape of labels:", y.shape)

# splitting the data into training and validation sets on the patient level
train_patients, val_patients = train_test_split(
    images['Patient ID'].unique(),
    test_size=0.2,
    random_state=1
)

train_df = images[images['Patient ID'].isin(train_patients)].reset_index(drop=True)
val_df = images[images['Patient ID'].isin(val_patients)].reset_index(drop=True)

# Encode labels
train_y = mlb.transform(train_df['Finding Labels'])
val_y = mlb.transform(val_df['Finding Labels'])

#dataset class


# data transfroms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])  
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Datasets and Dataloaders for training and validation
train_dataset = ChestXRayDataset(train_df, train_y, IMAGES_DIR, transform=train_transforms)
val_dataset = ChestXRayDataset(val_df, val_y, IMAGES_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# instantiate the model
model = create_model(num_classes=NUM_CLASSES)
model = model.to(device)

# loss and optimizer
criterion = nn.BCELoss()

# since we only taking a subset, we will calculate the class weights based on the training data so deal with any class imbalance
class_counts = train_y.sum(axis=0)
class_weights = 1. / (class_counts + 1e-5)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.BCELoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train and eval loop
best_val_loss = float('inf')


if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

    print("Training complete.")
