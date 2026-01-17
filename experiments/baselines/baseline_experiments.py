# src/experiments/baseline_experiments.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data.cl_benchmark_generator import CLBenchmarkGenerator
import timm
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import logging
import os
import random
import yaml

print("no lora test2")


def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return model

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def run_baseline_experiments_for_dataset(dataset_name, dataset_config, global_config):
    print(f"Running baseline experiments for dataset: {dataset_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
    
    # Get all unique domains
    domains = list(set(dataset.domains))
    num_domains = len(domains)
    print(f"Total number of domains: {num_domains}")

    # Randomly select test domain
    test_domain = random.choice(domains)
    train_domains = [d for d in domains if d != test_domain]
    print(f"Test domain: {test_domain}")
    print(f"Train domains: {train_domains}")

    # Create model
    model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(global_config['learning_rate']))

    # Baseline 1: Train and test on test domain only
    print("Baseline 1: Training and testing on test domain only")
    test_indices = [i for i, d in enumerate(dataset.domains) if d == test_domain]
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=global_config['batch_size'], shuffle=True)
    
    model = train(model, test_loader, criterion, optimizer, device, global_config['num_epochs'])
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Baseline 1 - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Reset model for Baseline 2
    model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
    model = model.to(device)

    # Baseline 2: Train on other domains with LoRA, then test on test domain
    print("Baseline 2: Training on other domains with LoRA, then testing on test domain")
    for train_domain in train_domains:
        print(f"Training on domain: {train_domain}")
        train_indices = [i for i, d in enumerate(dataset.domains) if d == train_domain]
        train_subset = Subset(dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)

        # Create LoRA adapter
        lora_config = LoraConfig(
            r=global_config['lora_r'],
            lora_alpha=global_config['lora_alpha'],
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=global_config['lora_dropout'],
          
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model = lora_model.to(device)

        # Train with LoRA
        optimizer = optim.AdamW(lora_model.parameters(), lr=float(global_config['learning_rate']))
        lora_model = train(lora_model, train_loader, criterion, optimizer, device, global_config['num_epochs'])

        # Merge LoRA weights
        model = lora_model.merge_and_unload()

    # Final evaluation on test domain
    print("Evaluating final model on test domain")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Baseline 2 - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save final model
    output_path = os.path.join(global_config['output_dir'], f'{dataset_name}_final_model.pth')
    torch.save(model.state_dict(), output_path)
    print(f"Final model for {dataset_name} saved to {output_path}")

    return {
        'dataset': dataset_name,
        'baseline1_accuracy': test_accuracy,
        'baseline2_accuracy': test_accuracy
    }

def main(config):
    results = []
    for dataset_name, dataset_config in config['datasets'].items():
        result = run_baseline_experiments_for_dataset(dataset_name, dataset_config, config)
        results.append(result)
    
    # Print summary of results
    print("\nSummary of Results:")
    for result in results:
        print(f"Dataset: {result['dataset']}")
        print(f"  Baseline 1 Accuracy: {result['baseline1_accuracy']:.2f}%")
        print(f"  Baseline 2 Accuracy: {result['baseline2_accuracy']:.2f}%")
        print("---")

if __name__ == "__main__":
    with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/baseline_configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)



