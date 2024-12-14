import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    # We will be using a pretrained model ResNet50 
    # ResNet50 model has already been trained on a large dataset ImageNet
    # it has learned general features like edges, textures, and shapes
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    '''
    The final fully connected layer of ResNet50 is designed for ImageNet 1000 classes. 
    We are replacing this layer with your own classification head which is configured for 
    the 15 classes (including "No Finding") in our X-ray dataset
    '''
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    
    return model
