from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

def get_model(model_name: str, device, lr: Optional[float] = None):
    if model_name not in ['ResNet18', 'ResNet50', 'Inception3']:
        raise NotImplementedError(f"No such model class exists... {(model_name)}")

    if model_name == 'ResNet50':
        models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)

    elif model_name == 'Inception3':
        model = models.inception_v3(pretrained=True)


    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # Define output layers
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=2048, out_features=200, bias=True)

    # Define loss criterion + optimizer
    criterion =  torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    return model, criterion, optimizer, scheduler, num_ftrs