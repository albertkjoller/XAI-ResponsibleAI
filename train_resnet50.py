from CUB.dataset import load_data
import torch
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import wandb


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="XAI",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "ResNet50",
    "dataset": "CUB_processed",
    "epochs": 100,
    },
    name="ResNet50_test"
)


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(in_features=2048, out_features=200, bias=True)
model.cuda()
compiled_model = torch.compile(model)

LR = 1e-4

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

NUM_EPOCHS = 100

train_loader = load_data(pkl_paths=['CUB_processed/class_attr_data_10/train.pkl'],use_attr=True,no_img=False,batch_size=32, resol=224)

val_loader = load_data(pkl_paths=['CUB_processed/class_attr_data_10/val.pkl'],use_attr=True,no_img=False,batch_size=32, resol=224)

scaler = torch.cuda.amp.GradScaler()

best_val_loss = torch.inf

print('Starting training')
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    for images, labels, _ in tqdm(train_loader):
        image_b = images.cuda(non_blocking=True)
        label = labels.cuda(non_blocking=True)
        
        ### YOUR CODE HERE ###
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = compiled_model(image_b)
            loss = loss_fn(output,label)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.detach()
        train_acc += sum(output.softmax(dim=1).argmax(dim=1) == label)
        
    
    train_loss = train_loss/len(train_loader)
    train_acc = train_acc/len(train_loader.dataset)
    
    wandb.log({"Epoch":epoch, "train/acc": train_acc, "train/loss": train_loss})
    
    print(f"Epoch {epoch} training loss: {train_loss}, accuracy: {train_acc}")
        

    model.eval()
    for images, labels, _ in tqdm(val_loader):
        image_b = images.cuda(non_blocking=True)
        label = labels.cuda(non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = compiled_model(image_b)
                loss = loss_fn(output,label)
                
            val_loss += loss.detach()
            val_acc += sum(output.softmax(dim=1).argmax(dim=1) == label)
            
    val_loss = val_loss/len(val_loader)
    val_acc = val_acc/len(val_loader.dataset)
    
    wandb.log({"Epoch":epoch, "val/acc": val_acc, "val/loss": val_loss})
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch} validation loss: {val_loss}, accuracy: {val_acc}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "ResNet50_best_model_checkpoint.pth")
        print(f"New best model at epoch {epoch}")    
