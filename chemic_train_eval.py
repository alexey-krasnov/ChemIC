"""
Train a deep learning model to use it for classification chemical images into one of four predefined classes:
- images with single chemical structure
- images with chemical reactions
- images multiple chemical structures
- images with no chemical structures

Usage Example:
You can run the script from the command line as follows:
python chemic_train_eval.py --dataset_dir /path/to/data --checkpoint_path /path/to/checkpoint.pth --models_dir /path/to/models
This will execute the training and evaluation using the specified paths.

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: September 18, 2023
"""

import argparse
import os
from pathlib import Path
import time
from datetime import timedelta, datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# Constants
NUM_EPOCHS = 100  # Set a large number of epochs
EARLY_STOPPING_PATIENCE = 20  # Number of consecutive epochs with no improvement to wait before stopping
BATCH_SIZE = 32
NUM_CLASSES = 4 # Number of classes for classification - one_molecule, reactions, several_molecules, rest
MODEL_NAME = "resnet50"  # define the models name


def main(args):
    start = time.time()
    # Move model and data to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Data Transformation
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),  # Convert to PIL Image
        v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data Transformation with Data Augmentation for train set
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),  # Randomly flip images horizontally
        v2.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        v2.ToImage(),  # Convert to PIL Image
        v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the dataset name according to name of the folder with images dataset, i.e. 'dataset_for_image_classifier'
    dataset_name = Path(args.dataset_dir).parent.name

    # Define datasets and dataloaders
    train_dataset = ImageFolder(root=os.path.join(args.dataset_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(args.dataset_dir, 'test'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(args.dataset_dir, 'validation'), transform=transform)

    n_cpu = os.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=n_cpu)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=n_cpu)

    # Define the model
    try:
        # Load your previously trained model
        checkpoint = torch.load(args.checkpoint_path)

        # Reinitialize the model with the modified architecture
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Adjust to output 4 classes

        # Load the weights from the checkpoint
        model.load_state_dict(checkpoint)

    except FileNotFoundError:
        print('Use pretrained models from Pytorch...')
        # If we haven't trained models, load pretrained from pytorch
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Adjust to output 4 classes

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # Add this line to initialize the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Create folder models including subfolder with corresponding dataset name for further trainings
    models_dir = f'{args.models_dir}/{dataset_name}'
    # Create folder models for further training
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Initialize variables to keep track of the best models and epoch
    no_improvement_count = 0
    best_val_accuracy = 0.0
    best_model = None
    best_epoch = 0

    # Training loop
    log_file_path = os.path.join(models_dir, f'chemical_image_classifier_{MODEL_NAME}_augmentation_{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.txt')
    with open(log_file_path, 'w') as log_file:
        for epoch in range(NUM_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader):.4f}", file=log_file)
            # Validation loop
            model.eval()
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            # Calculate validation accuracy after the entire validation loop
            val_accuracy = correct_val / total_val
            print(f"Validation Accuracy: {val_accuracy:.4f}", file=log_file)

            # Add this line to update the learning rate
            scheduler.step(val_accuracy)

            # Save variable the best models based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model.state_dict()
                best_epoch = epoch
                no_improvement_count = 0  # Reset the count since there's improvement
            else:
                no_improvement_count += 1

            # Check for early stopping
            if no_improvement_count >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy.", file=log_file)
                break  # Stop training

        print("Training complete.")
        log_file.write("Training complete.\n")

    # Save the best models at the latest epoch
    if best_model:
        print("Save best models...")
        best_model_name = os.path.join(models_dir, f'chemical_image_classifier_{MODEL_NAME}_{best_epoch}epochs_{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}.pth')
        torch.save(best_model, best_model_name)

        # Load the best models for testing
        # model.load_state_dict(torch.load(best_model_name))
        best_model.eval()
        correct_test = 0
        total_test = 0
        # Create lists to store true labels and predicted labels
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                # Collect predicted and true labels for later calculation of metrics
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        test_accuracy = correct_test / total_test
        with open(log_file_path, 'a') as log_file:
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}", file=log_file)

            # Convert the predicted and true labels to PyTorch tensors
            predicted_labels_tensor = torch.tensor(predicted_labels)
            true_labels_tensor = torch.tensor(true_labels)

            # Compute precision, recall, and F1-score using PyTorch functions
            accuracy = torchmetrics.functional.accuracy(predicted_labels_tensor, true_labels_tensor, num_classes=4, average='weighted', task='multiclass')
            precision = torchmetrics.functional.precision(predicted_labels_tensor, true_labels_tensor, num_classes=4, average='weighted', task='multiclass')
            recall = torchmetrics.functional.recall(predicted_labels_tensor, true_labels_tensor, num_classes=4, average='weighted', task='multiclass')
            f1 = torchmetrics.functional.f1_score(predicted_labels_tensor, true_labels_tensor, num_classes=4, average='weighted', task='multiclass')

            print(f"Accuracy: {accuracy:.4f}", file=log_file)
            print(f"Precision: {precision:.4f}", file=log_file)
            print(f"Recall: {recall:.4f}", file=log_file)
            print(f"F1-score: {f1:.4f}", file=log_file)

        class_names = train_dataset.classes
        # Extend the image_ids list with the image IDs from the current batch
        image_ids = [Path(image_info[0]).name for image_info in test_loader.dataset.imgs]
        # Convert the true prediction to class names
        true_names = [class_names[image_info[1]] for image_info in test_loader.dataset.imgs]
        predicted_names = [class_names[label] for label in predicted_labels]

        # Create a DataFrame to store the results
        df = pd.DataFrame({
            "image_id": image_ids,
            "true": true_names,
            "prediction": predicted_names
        })

        df.sort_values(by='image_id', inplace=True)

        df_name = os.path.join(models_dir, f"predictions_{MODEL_NAME}_{best_epoch}epochs_augmentation.csv")
        # Export the DataFrame to a CSV file
        df.to_csv(df_name, index=False)
        end = time.time()
        with open(log_file_path, 'a') as log_file:
            print(f'Whole process took {str(timedelta(seconds=(end-start)))} sec ', file=log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a ResNet model for Chemical Image Classification.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing the dataset (train, test, validation subdirectories)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the existed model checkpoint file')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save new trained models')

    args = parser.parse_args()
    main(args)
