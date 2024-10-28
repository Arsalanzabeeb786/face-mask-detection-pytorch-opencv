import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm






def train_model(model, train_loader, val_loader,criterion,optimizer,device, num_epochs ):

    """
    Trains and evaluates the model, with checkpointing based on the best validation accuracy.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function, e.g., `BCEWithLogitsLoss`.
        optimizer (torch.optim.Optimizer): The optimizer for model training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model. Default is 10.

    Returns:
        torch.nn.Module: The trained model.
    
    Example:
        trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)
    
    Notes:
        - This function trains the model using a loop for each epoch.
        - The model's performance on the validation dataset is evaluated at each epoch.
        - If the validation accuracy improves, the model is saved as a checkpoint.
        - The training and validation loss and accuracy are printed for each epoch.
    """

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0  # Variable to store the best validation accuracy

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        model.train()  # Ensure the model is in training mode
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.float().to(device)  # Convert labels to float for BCEWithLogitsLoss
            labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
            
            optimizer.zero_grad()

            # Forward pass - extract only logits if present
            outputs = model(inputs)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits

            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item() * inputs.size(0)

            # For BCEWithLogitsLoss, logits > 0 are predicted as class 1, otherwise class 0
            predicted = (logits > 0.0).float()  # Convert logits to predictions (0 or 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.float().to(device)  # Convert labels to float for BCEWithLogitsLoss
                labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]

                outputs = model(inputs)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits

                loss = criterion(logits, labels)

                # Accumulate validation loss
                val_loss += loss.item() * inputs.size(0)

                # Validation accuracy
                predicted = (logits > 0.0).float()  # Convert logits to predictions (0 or 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Save the best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving new best model with validation accuracy: {val_acc:.4f}")
            torch.save(model.state_dict(), f"models/val_accuracy_{val_acc:.4f}_model.pth")  # Fixed variable name

        # Append to lists for plotting
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accuracies.append(epoch_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    return model 


def test_model(model, test_loader):
    """
    Evaluates the model on the test dataset and calculates various metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        

    Returns:
        None

    Prints:
        - Test Accuracy
        - F1 Score
        - Precision
        - Recall
        - Classification report (for binary classification)
    
    Example:
        test_model(trained_model, test_loader, test_dataset)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Data"):
            inputs, labels = inputs.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)  # Reshape labels to [batch_size, 1]
            
            # Get model outputs (logits)
            outputs = model(inputs)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits

            # Convert logits to binary predictions (threshold 0)
            predicted = (logits > 0.0).float()

            # Collect true and predicted labels for further metrics
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Flatten the lists for metric calculations
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # F1-score, Precision, Recall for binary classification
    f1 = f1_score(y_true, y_pred)  # By default, for binary classification
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    # Classification Report (Detailed metrics for each class)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['WithMask', 'WithoutMask']))
   



