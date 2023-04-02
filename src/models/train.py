#!/usr/bin/python
#

import os
from pathlib2 import Path
import torch.nn.functional as F

import torch

import time
from tqdm import trange

from src.models.model import get_model
from src.models.utils import set_seed
from src.data.dataloader import get_loaders, get_loaders_with_concepts
from src.data.bottleneck_code.dataset import load_data
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
    name="ResNet50.100epochs.lr1e-4.bz32"
)

#  ---------------  Training  ---------------
def train(
        datafolder_path: str, datafile_name: str, bottleneck_loaders: bool,
        model_name: str,
        batch_size: int = 128, num_workers: int = 1, lr=1e-4, epochs: int = 100,
        experiment_name: str = str(int(round(time.time()))), save_path: str = '',
        seed: int = 42,
    ):

    # Set seed
    set_seed(seed)
    # Tensorboard writer for logging experiments

    train_loader = load_data(pkl_paths=['../data/src/bottleneck_code/CUB_processed/class_attr_data_10/train.pkl'],use_attr=False,no_img=False,batch_size=batch_size, resol=224)

    val_loader = load_data(pkl_paths=['../data/src/bottleneck_code/CUB_processed/class_attr_data_10/val.pkl'],use_attr=False,no_img=False,batch_size=batch_size, resol=224)

    scaler = torch.cuda.amp.GradScaler()

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"INFO - using device: {device}")

    # Define the model, loss criterion and optimizer
    model, loss_fn, optimizer, scheduler, _ = get_model(model_name, lr=lr, device=device)
    model.cuda()
    compiled_model = torch.compile(model)

    current_best_loss = torch.inf
    with trange(epochs) as t:
        for epoch in t:
            model.train()
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            for images, labels in tqdm(train_loader):
                image_b = images.cuda(non_blocking=True)
                label = labels.cuda(non_blocking=True)
                # Zero the parameter gradients
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
            
            model.eval()
            for images, labels in tqdm(val_loader):
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
            
            wandb.log({"Epoch":epoch, "val/acc": val_acc, "val/loss": val_loss, "train/acc": train_acc, "train/loss": train_loss})
            
            scheduler.step(val_loss)
            
            
            if val_loss < current_best_loss:
                current_best_loss = val_loss
                # Create and save checkpoint
                checkpoint = {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "model": {
                        'name': model_name,
                    },
                    "training_parameters": {
                        "save_path": save_path,
                        "lr": lr,
                        "optimizer": optimizer,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "device": device,
                    },
                    "data": {
                        "bottleneck_loader": bottleneck_loaders,
                        "filename": datafile_name
                    },
                    "best_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                }
                os.makedirs(f"{save_path}/{experiment_name}", exist_ok=True)
                torch.save(checkpoint, f"{save_path}/{experiment_name}/best.ckpt")


            # Update progress bar
            train_loss_descr = (
                f"Train loss: {train_loss:.3f}"
            )
            val_loss_descr = (
                f"Validation loss: {val_loss:.3f}"
            )
            train_acc_descr = (
                f"Train accuracy: {train_acc:.3f}"
            )
            val_acc_descr = (
                f"Validation accuracy: {val_acc:.3f}"
            )
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> {train_loss_descr} | {val_loss_descr} | {train_acc_descr} | {val_acc_descr} | Progress: "
            )

if __name__ == '__main__':

    # BASE_PATH = Path('projects/xai/XAI-ResponsibleAI')
    BASE_PATH = Path()

    datafolder_path = BASE_PATH / 'data'
    save_path = BASE_PATH / 'models'

    train(
        datafolder_path=datafolder_path,
        # datafile_name='03-24-2023-processed_data_224x224.pth',
        datafile_name='',
        bottleneck_loaders=True,
        model_name='ResNet50',
        batch_size=32,
        epochs=100,
        lr=1e-4,
        experiment_name='ResNet50.100epochs.lr1e-4.bz32',
        save_path=save_path,
    )