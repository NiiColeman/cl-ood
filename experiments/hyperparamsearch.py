# src/experiments/hyperparameter_search.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.cl_benchmark_generator import CLBenchmarkGenerator
import timm
from tqdm import tqdm
import itertools
import yaml
import os
# import logging

import os
import sys
import io

# Force flushing of stdout and stderr
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), 'wb', 0), write_through=True)

# Ensure the logs directory exists
# os.makedirs('logs', exist_ok=True)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs/lora_hyperparameter_search.log'),
#         logging.StreamHandler()  # This will also print logs to console
#     ]
# )

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets, _ in tqdm(train_loader, desc="Training", leave=False):
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        except RuntimeError as e:
            print(f"Error during training: {str(e)}")
            continue
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(val_loader, desc="Evaluating", leave=False):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            except RuntimeError as e:
                print(f"Error during evaluation: {str(e)}")
                continue
    return total_loss / len(val_loader), 100. * correct / total

def hyperparameter_search(dataset, config, device, dataset_name):
    num_classes = dataset.num_classes
    print(f"\nDataset: {dataset_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of domains: {dataset.num_domains}")
    print(f"Total number of samples: {len(dataset)}")

    # Create train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    learning_rates = config['learning_rates']
    batch_sizes = config['batch_sizes']
    num_epochs_list = config['num_epochs_list']

    best_accuracy = 0
    best_params = None
    best_model = None

    for lr, batch_size, num_epochs in itertools.product(learning_rates, batch_sizes, num_epochs_list):
        print(f"\nTrying: lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = timm.create_model(config['base_model'], pretrained=True, num_classes=num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=float(lr))

        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_params = {'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}
            best_model = model.state_dict()

    return best_params, best_accuracy, best_model

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets
    datasets = {
        name: CLBenchmarkGenerator(path, max_samples_per_class=config.get('max_samples_per_class'))
        for name, path in config['datasets'].items()
    }

    # Ensure the output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)

    # Perform hyperparameter search for each auxiliary dataset
    for dataset_name in config['auxiliary_datasets']:
        dataset = datasets[dataset_name]

        best_params, best_accuracy, best_model = hyperparameter_search(
            dataset, config, device, dataset_name
        )

        print(f"\nBest hyperparameters for {dataset_name}:")
        print(f"Best params: {best_params}")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")

        # Save the best model
        model_path = os.path.join(config['output_dir'], f"best_model_{dataset_name}.pth")
        torch.save(best_model, model_path)
        print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/custom_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)