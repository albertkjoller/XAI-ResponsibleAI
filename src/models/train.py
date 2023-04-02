#!/usr/bin/python
#

import os
from pathlib2 import Path
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torch

import time
from tqdm import trange

from src.models.model import get_model
from src.models.utils import set_seed
from src.data.dataloader import get_loaders, get_loaders_with_concepts
from src.data.bottleneck_code.dataset import load_data

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
    writer = SummaryWriter(f"logs/{experiment_name}")

    if not bottleneck_loaders:
        # Get dataset splits
        loaders, normalization = get_loaders(
            data_path=datafolder_path / 'processed/CUB_200_2011' / datafile_name,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    else:
        loaders, normalization = get_loaders_with_concepts(
            data_folder=datafolder_path,
            batch_size=batch_size,
        )

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"INFO - using device: {device}")

    # Define the model, loss criterion and optimizer
    model, criterion, optimizer, scheduler, _ = get_model(model_name, lr=lr, device=device)

    print("CNN Architecture:")
    print(model)

    current_best_loss = torch.inf
    with trange(epochs) as t:
        for epoch in t:
            running_loss_train, running_loss_val    = 0.0, 0.0
            running_acc_train,  running_acc_val     = 0.0, 0.0

            for batch in iter(loaders['train']):
                # Extract data
                if bottleneck_loaders:
                    inputs, labels, concepts = batch
                    inputs, labels, concepts = inputs.to(device), labels.to(device), torch.stack(concepts).T.to(device)
                else:
                    inputs, labels = batch['image'].to(device), batch['label'].to(device)

                if model_name == 'Inception3' and not bottleneck_loaders:
                    # Resize the input tensor to a larger size
                    inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=True)

                # Zero the parameter gradients
                optimizer.zero_grad()
                model.to(device)

                # Forward + backward
                outputs = model(inputs).logits if model_name == 'Inception3' else model(inputs)

                loss = criterion(outputs, labels)
                running_loss_train += loss.item()
                loss.backward()
                # Optimize
                optimizer.step()
                scheduler.step()

                # Get predictions from log-softmax scores
                preds = torch.exp(outputs.detach()).topk(1)[1]
                # Store accuracy
                equals = preds.flatten() == labels
                running_acc_train += torch.mean(equals.type(torch.FloatTensor))

            # Validation
            with torch.no_grad():
                for batch in iter(loaders['validation']):
                    # Extract data
                    if bottleneck_loaders:
                        inputs, labels, concepts = batch
                        inputs, labels, concepts = inputs.to(device), labels.to(device), torch.stack(concepts).T.to(device)
                    else:
                        inputs, labels = batch['image'].to(device), batch['label'].to(device)

                    if model_name == 'Inception3' and not bottleneck_loaders:
                        # Resize the input tensor to a larger size
                        inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=True)

                    # Forward + backward
                    outputs = model(inputs).logits if model_name == 'Inception3' else model(inputs)
                    preds = torch.exp(outputs).topk(1)[1]

                    # Compute loss and accuracy
                    running_loss_val += criterion(outputs, labels)
                    equals = preds.flatten() == labels
                    running_acc_val += torch.mean(equals.type(torch.FloatTensor))

            if running_loss_val / len(loaders['validation']) < current_best_loss:
                current_best_loss = running_loss_val / len(loaders['validation'])
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
                        "filename": datafile_name,
                        "normalization": {
                            "mu": list(normalization['mean'].numpy()),
                            "sigma": list(normalization['std'].numpy()),
                        },
                    },
                    "best_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                }
                os.makedirs(f"{save_path}/{experiment_name}", exist_ok=True)
                torch.save(checkpoint, f"{save_path}/{experiment_name}/best.ckpt")


            # Update progress bar
            train_loss_descr = (
                f"Train loss: {running_loss_train / len(loaders['train']):.3f}"
            )
            val_loss_descr = (
                f"Validation loss: {running_loss_val / len(loaders['validation']):.3f}"
            )
            train_acc_descr = (
                f"Train accuracy: {running_acc_train / len(loaders['train']):.3f}"
            )
            val_acc_descr = (
                f"Validation accuracy: {running_acc_val / len(loaders['validation']):.3f}"
            )
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> {train_loss_descr} | {val_loss_descr} | {train_acc_descr} | {val_acc_descr} | Progress: "
            )

            writer.add_scalar('loss/train', running_loss_train / len(loaders['train']), epoch)
            writer.add_scalar('accuracy/train', running_acc_train / len(loaders['train']), epoch)
            writer.add_scalar('loss/validation', running_loss_val / len(loaders['validation']), epoch)
            writer.add_scalar('accuracy/validation', running_acc_val / len(loaders['validation']), epoch)

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
        model_name='Inception3',
        batch_size=32,
        epochs=100,
        lr=0.0045,
        experiment_name='Inception3.100epochs.lr0.0045.bz32',
        save_path=save_path,
    )