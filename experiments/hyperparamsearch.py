# src/experiments/hyperparameter_search.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from data.cl_benchmark_generator import CLBenchmarkGenerator
import timm
from tqdm import tqdm
import itertools
import yaml

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets, _ in tqdm(train_loader, desc="Training", leave=False):
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
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(test_loader), 100. * correct / total

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets
    datasets = {
        name: CLBenchmarkGenerator(path, max_samples_per_class=config.get('max_samples_per_class'))
        for name, path in config['datasets'].items()
    }

    # Combine auxiliary datasets for training
    train_datasets = [datasets[name] for name in config['auxiliary_datasets']]
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = datasets[config['test_dataset']]

    # Determine the total number of unique classes across all datasets
    all_classes = set()
    for dataset in datasets.values():
        all_classes.update(dataset.class_to_idx.keys())
    num_classes = len(all_classes)
    print(f"Total number of unique classes: {num_classes}")

    # Create a global class mapping
    global_class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}

    # Update the class indices in all datasets
    for dataset in datasets.values():
        dataset.update_class_indices(global_class_to_idx)

    # Hyperparameters to search
    learning_rates = config['learning_rates']
    batch_sizes = config['batch_sizes']
    num_epochs_list = config['num_epochs_list']

    best_accuracy = 0
    best_params = None

    for lr, batch_size, num_epochs in itertools.product(learning_rates, batch_sizes, num_epochs_list):
        print(f"\nTrying: lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = timm.create_model(config['base_model'], pretrained=True, num_classes=num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=float(lr))

        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_params = {'lr': lr, 'batch_size': batch_size, 'num_epochs': num_epochs}
            torch.save(model.state_dict(), config['best_model_path'])

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best test accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/custom_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)