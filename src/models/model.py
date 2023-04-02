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
        model = models.resnet50(pretrained=True)

    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)

    elif model_name == 'Inception3':
        model = models.inception_v3(pretrained=True)


    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # Define output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 200),
        nn.LogSoftmax(dim=1)
    ).to(device)

    # Define loss criterion + optimizer --> NLLLoss used with LogSoftmax for stability reasons
    criterion = nn.NLLLoss()

    if model_name == 'Inception3':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.94)
    return model, criterion, optimizer, scheduler, num_ftrs