# Reimplementation of OR-UNet for binary 
# tool segmentation for the robustmis dataset
# Paper: https://arxiv.org/pdf/2004.12668.pdf

import torch
import numpy as np
import random
import argparse
import os

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop

from data import prepare_cross_validation_datasets, AdaptiveResize, StatefulRandomCrop
from model import OrUnet
from loss import OrUnetLoss

def train(out_dir, cross_validation_index):

    SEED = 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    out_dir = f"{out_dir}/cross_validation_{cross_validation_index}"

    print(f"Training cross validation model {cross_validation_index}.")
    print(f"Saving output to ''{out_dir}''.")
    
    os.makedirs(out_dir)

    device = "cuda"
    batch_size = 16

    # Data Loader:
    training_transform = Compose([AdaptiveResize((270, 480)), StatefulRandomCrop((256, 448))])
    validation_transform = Compose([AdaptiveResize((270, 480)), CenterCrop((256, 448))])
    training_dataset, validation_dataset = prepare_cross_validation_datasets("robustmislite", cross_validation_index, training_transform, validation_transform)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=batch_size//2, pin_memory=True, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=batch_size//2, pin_memory=True)

    # Network Architecture:
    model = OrUnet().to(device=device)

    # Loss and Optimizer:
    loss_function = OrUnetLoss()
    optimizer = SGD(model.parameters(), lr=1.0, momentum=0.9, nesterov=True)
    scheduler = ExponentialLR(optimizer, 0.9)

    # Early Stopping:
    max_epochs = 2000
    early_stopping_patience = 25
    early_stopping_counter = 0
    best_validation_loss = 1e9

    training_losses = []
    validation_losses = []

    for epoch in range(max_epochs):
        print(f"Epoch {epoch} of {max_epochs}")

        # Training Loop:
        model.train()
        loss_sum = 0.0
        norm = 0.0

        for input, target in training_dataloader:

            input, target = input.to(device=device), target.to(device=device)

            optimizer.zero_grad()
            output = model(input)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            loss_sum += loss * input.shape[0]
            norm += input.shape[0]

        scheduler.step()
        training_loss = loss_sum / norm
        training_losses.append(training_loss)

        # Validation Loop:
        model.eval()
        loss_sum = 0.0
        norm = 0.0

        with torch.no_grad():
            for input, target in validation_dataloader:

                input, target = input.to(device=device), target.to(device=device)

                output = model(input)
                loss = loss_function(output, target)

                loss_sum += loss * input.shape[0]
                norm += input.shape[0]

        validation_loss = loss_sum / norm
        validation_losses.append(validation_loss)

        # Early Stopping Update:
        if validation_loss < best_validation_loss:
            early_stopping_counter = 0
            best_validation_loss = validation_loss
            torch.jit.script(model).save(f"{out_dir}/model.pt")
        else:
            early_stopping_counter += 1

        # Saving losses:
        torch.save(training_losses, f"{out_dir}/training_losses.pt")
        torch.save(validation_losses, f"{out_dir}/validation_losses.pt")

        # Epoch Report:
        print(f"Training Loss:        {training_loss:.4f}")
        print(f"Validation Loss:      {validation_loss:.4f}")
        print(f"Best Validation Loss: {best_validation_loss:.4f}")
        print(f"Early Stopping:       {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter == early_stopping_patience:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-directory", help="Directory to save the training output")
    parser.add_argument("-i", "--cross-validation-index", help="The index of the surgery to use for validation (0-7)", type=int)
    args = parser.parse_args()

    if args.cross_validation_index < 0 or args.cross_validation_index > 7:
        raise argparse.ArgumentTypeError("cross-validation-index is not in the range 0-7")

    train(args.output_directory, args.cross_validation_index)