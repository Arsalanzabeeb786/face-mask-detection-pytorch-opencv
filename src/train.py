"""
Main function to train and evaluate a MobileNetV3 model using command-line arguments for hyperparameters and dataset paths.

Command-Line Arguments:
    --lr (float): Learning rate for the optimizer (default: 0.0001).
    --epochs (int): Number of epochs for training (default: 10).
    --batch_size (int): Batch size for DataLoader (default: 32).
    --seed (int): Random seed for reproducibility (default: 42).
    --train_dir (str): Path to the training dataset.
    --val_dir (str): Path to the validation dataset.
    --test_dir (str): Path to the test dataset.

Example:
    To train the model with a learning rate of 0.001, batch size of 64, for 20 epochs, using custom dataset paths:
    
    python train.py --lr 0.001 --epochs 10  --batch_size 64 --seed 42 
    --train_dir "dataset/Train" --val_dir "dataset/Validation" --test_dir "dataset/Test"

    This would:
    - Train the model for 20 epochs with a learning rate of 0.001 and a batch size of 64.
    - Set the random seed to 42 for reproducibility.
    - Use the specified directories for the training, validation, and test datasets.
"""



import argparse
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm
import multiprocessing
import os
from torchinfo import summary
import dataloaders, engine, model_builder

import random
import numpy as np

def set_seed(seed_value=42):
    """
    Sets seed for reproducibility across various libraries and ensures deterministic behavior.

    Args:
        seed_value (int): The seed value to use for random number generators.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parses command-line arguments for training configuration, including paths and hyperparameters.

    Returns:
        argparse.Namespace: Parsed arguments including learning rate, epochs, batch size, seed, and dataset paths.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a MobileNetV3 model.")

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Add dataset paths as arguments
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation dataset')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset')

    return parser.parse_args()

def main():
    """
    Main function to train and evaluate the model using command-line parameters for hyperparameters and dataset paths.
    """
    # Parse command-line arguments
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set the device for model training and evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Creating Model  .... ")
    # Load the pre-trained model and corresponding transforms
    model, transform = model_builder.load_model(device)

    print("[INFO] Reading dataset from provided dataset folders.... ")

    print("[INFO] Making train, validation, and test DataLoaders.... ")
    # Create the DataLoaders for training, validation, and testing using paths from argparse
    train_loader, val_loader, test_loader, class_names, class_idx = dataloaders.dataloaders(
        transform=transform, 
        batch_size=args.batch_size, 
        train_dir=args.train_dir, 
        val_dir=args.val_dir, 
        test_dir=args.test_dir
    )

    print("[INFO] Train, validation, and test DataLoaders are generated. Here are the details .... ")
    print(train_loader)
    print(val_loader)
    print(test_loader)
    print("Class Names: ", class_names)
    print("Class Index: ", class_idx)

    print("-----------------------------------------")
    print("[INFO] Starting Model Training .... ")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model using the engine's train_model function
    trained_model = engine.train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs)

    print("[INFO] Testing model on test dataset now ")
    # Test the trained model using the test DataLoader
    engine.test_model(trained_model, test_loader)

if __name__ == "__main__":
    main()

