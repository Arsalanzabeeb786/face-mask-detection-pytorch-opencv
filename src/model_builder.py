import torch
import torchvision
from torchvision import models, transforms
from torch import nn
from torchinfo import summary

def load_model(device):
    """
    Loads the MobileNetV3_Large model with pre-trained weights and prepares it for binary classification.

    This function performs the following steps:
    1. Loads the MobileNetV3_Large model with default pre-trained weights.
    2. Freezes all layers except for the last classifier and the final deep layers to allow fine-tuning.
    3. Modifies the classifier to output a single logit for binary classification (face mask detection).
    4. Moves the model to the specified device (GPU or CPU).
    5. Defines a transformation pipeline suitable for MobileNetV3.

    Args:
        device (torch.device): The device (CPU or GPU) to move the model to.

    Returns:
        model (torch.nn.Module): The MobileNetV3_Large model modified for binary classification.
        transform (torchvision.transforms.Compose): The transformation pipeline for preprocessing the dataset.

    Example:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, transform = load_model(device)
    """
    # Load MobileNetV3_Large with default pre-trained weights
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # Freeze all layers except for the classifier layer and the final deep layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier to output 1 logit for binary classification
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)

    # Unfreeze only the classifier and the last few layers for fine-tuning
    for param in model.features[12:].parameters():  # Unfreeze deeper layers
        param.requires_grad = True

    for param in model.classifier.parameters():  # Classifier layer remains trainable
        param.requires_grad = True

    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)

    # Define the transformation pipeline for MobileNetV3
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    # Print model summary using torchinfo
    print("[INFO] Model created. Here is model summary .... ")
    summary(model=model, 
            input_size=(64, 3, 224, 224),  # Batch size of 64, 3 channels, 224x224 input
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    return model, transform
