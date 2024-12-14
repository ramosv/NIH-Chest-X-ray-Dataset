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
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


DATA_CSV = "Data/Data_Entry_2017_v2020.csv"
IMAGES_DIR = "Data/images/"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
IMAGE_SIZE = 1024
NUM_CLASSES = 15

device = torch.device('cuda')
print(f'Using device: {device}')

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


mlb = MultiLabelBinarizer(classes=[
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
])

images['Finding Labels'] = images['Finding Labels'].str.split('|')
y = mlb.fit_transform(images['Finding Labels'])

print("Classes:", mlb.classes_)
print("Shape of labels:", y.shape)

train_patients, val_patients = train_test_split(
    images['Patient ID'].unique(),
    test_size=0.2,
    random_state=1
)

train_df = images[images['Patient ID'].isin(train_patients)].reset_index(drop=True)
val_df = images[images['Patient ID'].isin(val_patients)].reset_index(drop=True)

train_y = mlb.transform(train_df['Finding Labels'])
val_y = mlb.transform(val_df['Finding Labels'])

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

train_dataset = ChestXRayDataset(train_df, train_y, IMAGES_DIR, transform=train_transforms)
val_dataset = ChestXRayDataset(val_df, val_y, IMAGES_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = create_model(num_classes=NUM_CLASSES)
model = model.to(device)

class_counts = train_y.sum(axis=0)
class_weights = 1. / (class_counts + 1e-5)
class_weights = torch.FloatTensor(class_weights).to(device)
print("Class weights:", class_weights)

criterion = nn.BCEWithLogitsLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(log_dir='runs/ChestXRay_Experiment')

best_val_loss = float('inf')
scaler = GradScaler(enabled=True) 

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        optimizer.zero_grad()
        with autocast('cuda'): 
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {epoch_loss:.9f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_val, labels_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
            images_val, labels_val = images_val.to(device), labels_val.to(device)
            with autocast('cuda'):
                outputs_val = model(images_val)
                loss_val = criterion(outputs_val, labels_val)
            val_loss += loss_val.item() * images_val.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {val_loss:.9f}")

    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training complete.")
writer.close()

