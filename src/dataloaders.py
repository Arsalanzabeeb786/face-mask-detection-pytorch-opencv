import multiprocessing
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def dataloaders(transform, batch_size, train_dir, val_dir, test_dir):
    """
    Creates DataLoaders for training, validation, and testing datasets with optimal settings.

    Args:
        transform (torchvision.transforms.Compose): Transformations to be applied to the datasets (training, validation, and test).
        batch_size (int): Batch size for the DataLoaders.
        train_dir (str): Path to the training dataset.
        val_dir (str): Path to the validation dataset.
        test_dir (str): Path to the test dataset.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            - class_names (list): List of class names in the dataset.
            - class_idx (dict): Dictionary mapping class names to class indices.
    
    Example:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        batch_size = 32
        
        train_loader, val_loader, test_loader, class_names, class_idx = dataloaders(
            transform, batch_size, "path/to/train", "path/to/val", "path/to/test"
        )
    
    Notes:
        - The DataLoader uses all available CPU cores for optimal performance, determined by `multiprocessing.cpu_count()`.
        - `pin_memory=True` is used to speed up data transfer to the GPU when using CUDA.
        - `prefetch_factor=4` is set for the training DataLoader to fetch batches in advance, improving performance.
        - The validation and test DataLoaders are not shuffled, since data order is not required for these datasets.
    """
    
    # Load datasets using ImageFolder
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    
    # Extract class names and indices from the training dataset
    class_names = train_dataset.classes
    class_idx = train_dataset.class_to_idx
    
    # Get the number of available CPU cores
    num_workers = multiprocessing.cpu_count()   
    
    # Create the DataLoader for the training dataset
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Use multiple workers for data loading
        pin_memory=True,  # Use pinned memory for faster GPU transfer
        prefetch_factor=4,  # Prefetch batches for efficiency
        shuffle=True,  # Shuffle the dataset for training
    )
    
    # Create the DataLoader for the validation dataset
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle the validation set
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create the DataLoader for the test dataset
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle the test set
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names, class_idx
