import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

def ResNet(num_classes=8, pretrained=True):
    if pretrained:
        net = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    else:
        net = torch.hub.load("pytorch/vision", "resnet50")
    final_in_ftrs = net.fc.in_features
    net.fc = nn.Linear(final_in_ftrs, num_classes)
    return net

class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + np.finfo(np.float32).eps)
        else:
            loss = 0
        return loss
    
def compute_confusion_matrix(model, test_loader, device, model_name):
    model.eval()

    # Create empty lists to store the true labels and predicted labels
    test_labels = []
    test_pred_labels = []

    # Loop through the test set
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            # Get predictions
            if model_name=="DDAMFN":
                outputs, _, _ = model(batch_images)
            else:
                outputs = model(batch_images)

            _, predicted = torch.max(outputs, 1)

            # Append the true and predicted labels
            test_labels.append(batch_labels.cpu().numpy())
            test_pred_labels.append(predicted.cpu().numpy())

    # Flatten the lists
    test_labels = np.concatenate(test_labels)
    test_pred_labels = np.concatenate(test_pred_labels)

    # Create confusion matrix
    confusion_mtx = confusion_matrix(test_labels, test_pred_labels)
    class_names = test_loader.dataset.classes

    return confusion_mtx, class_names

# Custom function to track learning rate history
class LearningRateHistory:
    def __init__(self):
        self.lr_history = []

    def on_epoch_end(self, optimizer):
        self.lr_history.append(optimizer.param_groups[0]['lr'])

# Helper functions for plotting (same as before)
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')

    plt.show()


def plot_lr_history(lr_history, model, test_loader, device, model_name):
    lr = [lr_history[0]] + lr_history

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.step(np.arange(len(lr)), lr, where="post")
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Log(Learning Rate)')
    plt.title('Learning Rate vs Epochs')
    #plt.grid(True)

    plt.subplot(1, 2, 2)
    confusion_mtx, class_names = compute_confusion_matrix(model, test_loader, device, model_name)
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def train_FER(model, learning_rate, num_epochs, weight_path, train_loader, val_loader, test_loader, device, model_name, verbose=False, patience=100):    
    model.to(device)
    # Set up the learning rate scheduler (Exponential Decay)
    initial_learning_rate = learning_rate  # Save the initial learning rate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    if model_name=="DDAMFN":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss()  # Categorical crossentropy equivalent in PyTorch
    criterion_at = AttentionLoss()

    lr_history_callback = LearningRateHistory()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0
    patience_counter = 0

    # Initialize validation metrics to desired values
    avg_val_loss = float('inf') 
    val_accuracy = 0.0 

    # Create a single progress bar for the entire training
    pbar = tqdm(total=num_epochs, desc="Training Progress")

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Track the number of batches
        num_batches = len(train_loader)

        if model_name=="DDAMFN":
            # Reset learning rate after epoch 40
            if epoch%40==0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_learning_rate  # Reset learning rate to initial value
        
        # Loop over the batches
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if model_name=="DDAMFN":
                outputs,feat,heads = model(images)
                loss = criterion(outputs,labels)  + 0.1*criterion_at(heads)
                loss.backward()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Calculate average loss and accuracy for the current batch
            avg_train_loss = running_train_loss / (i+1)
            train_accuracy = 100 * correct_train / total_train

            # Update the progress bar description during training
            pbar.set_postfix(
                train_loss=f"{avg_train_loss:.4f}",
                train_acc=f"{train_accuracy:.2f}%",
                val_loss=f"{avg_val_loss:.4f}",
                val_acc=f"{val_accuracy:.2f}%")

        # After finishing the training loop for the epoch
        avg_train_loss = running_train_loss / num_batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(100 * correct_train / total_train)

        # Validation loop
        model.eval()  # Switch to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if model_name=="DDAMFN":
                    outputs,feat,heads = model(images)
                    loss = criterion(outputs,labels)  + 0.1*criterion_at(heads)
                else: 
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # Update the progress bar with validation metrics after the validation loop
        pbar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            train_acc=f"{train_accuracy:.2f}%",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{val_accuracy:.2f}%")
        
        # Store validation results
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Update learning rate schedule
        lr_scheduler.step()
        lr_history_callback.on_epoch_end(optimizer)

        # Early Stopping and model saving on best validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(weight_path, "best_model.pth"))
        else:
            patience_counter += 1
        
        if patience_counter > patience:
            #print("Stopping early due to lack of improvement in validation accuracy.")
            break

        # Update the progress bar after the epoch is done
        pbar.update(1)

    # Close the progress bar at the end of training
    pbar.close()

    # Store history in a dataframe
    history_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,})
    
    model.load_state_dict(torch.load(os.path.join(weight_path, "best_model.pth"), map_location=device))

    # Optionally plot the training history and the learning rate history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    learning_rate_history = lr_history_callback.lr_history
    plot_lr_history(learning_rate_history, model, test_loader, device, model_name)
    
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if model_name=="DDAMFN":
                outputs,feat,heads = model(images)
                loss = criterion(outputs,labels)  + 0.1*criterion_at(heads)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test

    # Return the history dataframe with test results
    history_df['test_loss'] = [avg_test_loss] * len(history_df)
    history_df['test_accuracy'] = [test_accuracy] * len(history_df)

    return history_df