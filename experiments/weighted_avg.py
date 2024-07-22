
"""
Baseline Experiments for Out-of-Distribution Generalization

This script implements three baseline experiments to evaluate out-of-distribution
generalization in image classification tasks using the PACS, VLCS, DomainNet, 
and Office-Home datasets.

Baselines:
1. Single Domain: Train and test on the target domain only.
2. Multi-Domain with LoRA: Train on source domains using LoRA adapters, then test on target domain.
3. Weighted LoRA Adapters: Combine LoRA adapters from source domains using learned weights.

Usage:
    python src/experiments/baseline_experiments.py

Configuration:
    The experiment parameters are specified in 'configs/baseline_config.yaml'.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from data.cl_benchmark_generator import CLBenchmarkGenerator
import timm
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import random
import os
import json 


class WeightedLoRAAdapter(nn.Module):
    def __init__(self, base_model, adapters):
        super().__init__()
        self.base_model = base_model
        self.adapters = nn.ModuleList(adapters)
        self.weights = nn.Parameter(torch.ones(len(adapters)) / len(adapters))

    def forward(self, x):
        base_output = self.base_model(x)
        adapter_outputs = [adapter(x) for adapter in self.adapters]
        weighted_output = sum(w * out for w, out in zip(self.weights, adapter_outputs))
        return base_output + weighted_output

def train(model, train_loader, criterion, optimizer, device, desc="Training"):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels, _ in tqdm(train_loader, desc=desc):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(test_loader), 100. * correct / total


def run_baseline_experiments(config):
    """
    Run all three baseline experiments using domain-incremental learning.

    This function orchestrates the execution of:
    Baseline 1: Train and test on a single domain of the test dataset.
    Baseline 2: Train on domains from auxiliary datasets, test on a held-out domain.
    Baseline 3: Weighted average of LoRA adapters, tested on a held-out domain.

    Args:
        config (dict): Configuration parameters for the experiments.

    Returns:
        dict: Accuracy results for all three baselines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets
    datasets = {}
    for name, dataset_config in config['datasets'].items():
        print(f"Loading dataset: {name}")
        datasets[name] = CLBenchmarkGenerator(
            dataset_config['path'],
            max_samples_per_class=config.get('max_samples_per_class')
        )

    # Select test dataset and domain
    test_dataset_name = config['test_dataset']
    test_dataset = datasets[test_dataset_name]
    test_domain = random.choice(range(test_dataset.num_domains))
    print(f"Selected test dataset: {test_dataset_name}, test domain: {test_domain}")

    # Create model
    num_classes = test_dataset.num_classes
    base_model = timm.create_model(config['base_model'], pretrained=True, num_classes=num_classes)
    base_model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Baseline 1: Train and test on single domain of test dataset
    print("Baseline 1: Training and testing on single domain of test dataset")
    test_domain_data = test_dataset.get_domain_data(test_domain)
    train_size = int(0.8 * len(test_domain_data))
    train_subset, test_subset = random_split(test_domain_data, [train_size, len(test_domain_data) - train_size])
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)

    optimizer = optim.AdamW(base_model.parameters(), lr=float(config['learning_rate']))
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train(base_model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    baseline_1_loss, baseline_1_accuracy = evaluate(base_model, test_loader, criterion, device)
    print(f"Baseline 1 - Test Loss: {baseline_1_loss:.4f}, Test Accuracy: {baseline_1_accuracy:.2f}%")

    # Reset model for Baseline 2 and 3
    base_model = timm.create_model(config['base_model'], pretrained=True, num_classes=num_classes)
    base_model = base_model.to(device)

    # Baseline 2: Train on domains from auxiliary datasets, test on held-out domain
    print("Baseline 2: Training on domains from auxiliary datasets, testing on held-out domain")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["query", "value"],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type=TaskType.IMAGE_CLASSIFICATION
    )

    adapters = []
    for dataset_name in config['auxiliary_datasets']:
        dataset = datasets[dataset_name]
        for domain in range(dataset.num_domains):
            print(f"Training on dataset: {dataset_name}, domain: {domain}")
            domain_data = dataset.get_domain_data(domain)
            train_loader = DataLoader(domain_data, batch_size=config['batch_size'], shuffle=True)

            lora_model = get_peft_model(base_model, lora_config)
            lora_model = lora_model.to(device)
            optimizer = optim.AdamW(lora_model.parameters(), lr=float(config['learning_rate']))
            
            for epoch in range(config['num_epochs']):
                train_loss, train_acc = train(lora_model, train_loader, criterion, optimizer, device)
                print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            adapters.append(lora_model)
            base_model = lora_model.merge_and_unload()

    # Evaluate Baseline 2
    baseline_2_loss, baseline_2_accuracy = evaluate(base_model, test_loader, criterion, device)
    print(f"Baseline 2 - Test Loss: {baseline_2_loss:.4f}, Test Accuracy: {baseline_2_accuracy:.2f}%")

    # Baseline 3: Weighted Average of LoRA Adapters
    print("Baseline 3: Weighted Average of LoRA Adapters")
    weighted_adapter = WeightedLoRAAdapter(base_model, adapters)
    weighted_adapter = weighted_adapter.to(device)
    optimizer = optim.Adam([weighted_adapter.weights], lr=float(config['weighted_adapter_lr']))

    for epoch in range(config['weighted_adapter_epochs']):
        train_loss, train_acc = train(weighted_adapter, train_loader, criterion, optimizer, device, desc=f"Baseline 3 Epoch {epoch+1}")
        print(f"Baseline 3 Epoch {epoch+1}/{config['weighted_adapter_epochs']}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Evaluate Baseline 3
    baseline_3_loss, baseline_3_accuracy = evaluate(weighted_adapter, test_loader, criterion, device)
    print(f"Baseline 3 - Test Loss: {baseline_3_loss:.4f}, Test Accuracy: {baseline_3_accuracy:.2f}%")

    # Save final models
    torch.save(base_model.state_dict(), os.path.join(config['output_dir'], 'baseline_2_final_model.pth'))
    torch.save(weighted_adapter.state_dict(), os.path.join(config['output_dir'], 'baseline_3_final_model.pth'))

    return {
        "baseline_1_accuracy": baseline_1_accuracy,
        "baseline_2_accuracy": baseline_2_accuracy,
        "baseline_3_accuracy": baseline_3_accuracy
    }




if __name__ == "__main__":
    import yaml
    import json
    import traceback

    try:
        # Load configuration
        with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/baseline_configs.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # If auxiliary_datasets is not specified, use all datasets except the test dataset
        if 'auxiliary_datasets' not in config:
            config['auxiliary_datasets'] = list(config['datasets'].keys())
            if 'test_dataset' in config:
                config['auxiliary_datasets'].remove(config['test_dataset'])
            print(f"No auxiliary_datasets specified. Using {config['auxiliary_datasets']} as auxiliary datasets.")

        # Run experiments
        results = run_baseline_experiments(config)
        
        # Print final results
        print("\nFinal Results:")
        print(f"Baseline 1 Accuracy: {results['baseline_1_accuracy']:.2f}%")
        print(f"Baseline 2 Accuracy: {results['baseline_2_accuracy']:.2f}%")
        print(f"Baseline 3 Accuracy: {results['baseline_3_accuracy']:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()