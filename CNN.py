import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    # We wil be using a pretrained model ResNet50 
    # ResNet50 model has already been trained on a large dataset ImageNet
    # it has learned general features like edges, textures, and shapes
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    

    '''
    The final fully connected layer of ResNet50 is designed for ImageNet 1000 classes. 
    We are replacing this layer with your own classification head which is configured for 
    the 14 diseases our x-ray your dataset
    '''
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),
        nn.Sigmoid() 
    )
    
    return model
