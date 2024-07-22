# src/experiments/baseline_experiments.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from cl_benchmark_generator import CLBenchmarkGenerator
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
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = global_config['device']
    print(f"Using device: {device}")

    # Load the dataset
    dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
    
    # Get all unique domains
    domains = list(set(dataset.domains))
    num_domains = len(domains)
    print(f"Total number of domains: {num_domains}")

    # The test domain is selected from the config file
    test_domain = dataset_config['test_domain']
    train_domains = [d for d in domains if d != test_domain]
    print(f"Test domain: {test_domain}")
    print(f"Train domains: {train_domains}")

    # Create model
    model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(global_config['learning_rate']))

    # Merging_LoRAs Baseline
    # Train (or load from memory) an adapter for each train domain
    # Merge with different algorithms the adapters, to get a single adapter
    # Merge the resulting adapter to the base model -> zero-shot test on the target domain

    # 1. Train (or load) an adapter for each train domain
    for train_domain in train_domains:
        # load the adapter from memory, if path is provided
        ada_path = global_config.get('saved_adapters', {}).get(dataset_name, {}).get(train_domain, None)
        if ada_path:
            print(f'Loading the adapter for {train_domain} domain from {ada_path}')
            model.load_adapter(ada_path)

        else: # train the adapter
            print(f'Training on domain: {train_domain}')
            train_indices = [i for i, d in enumerate(dataset.domains) if d == train_domain]
            train_subset = Subset(dataset, train_indices)
            train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)

            # Create LoRA adapter
            lora_config = LoraConfig(
                r = global_config['lora_r'],
                lora_alpha = global_config['lora_alpha'],
                target_modules = ["qkv", "fc1", "fc2"],
                lora_dropout = global_config['lora_dropout'],)
            
            # the variable 'lora_model' do not exists -> first time an adapter gets attached to the model
            # hence, we need to create the PeftModel
            if not 'lora_model' in locals():
                lora_model = get_peft_model(model, lora_config, adapter_name=f'{dataset_name}_{train_domain}_lora')

            # else, if the model has already some adapter, simply add a new one
            else:
                assert isinstance(lora_model, PeftModel), 'Your lora_model is not a PeftModel'
                lora_model.add_adapter(f'{dataset_name}_{train_domain}_lora', lora_config)
                lora_model.set_adapter(f'{dataset_name}_{train_domain}_lora')

            lora_model = lora_model.to(device)

            # Train with LoRA
            optimizer = optim.AdamW(lora_model.parameters(), lr=float(global_config['learning_rate']))
            lora_model = train(lora_model, train_loader, criterion, optimizer, device, global_config['num_epochs'])
            lora_model.save_pretrained(f'/leonardo_scratch/fast/IscrC_FoundCL/projects/cl-collab/ModelRatatouille/lquarant/cl-ood/saved_models/{dataset_name}_{train_domain}')

    # 2. Merge LoRAs with different algorithms & merge the final adapter to the model
    adapters = [f'{dataset_name}_{train_domain}_lora' for train_domain in train_domains]
    adapter_name = 'merge'
    combination_type = global_config['combination_type']
    weights = [1.0 for _ in range(len(adapters))]

    lora_model.add_weighted_adapter(adapters, weights, adapter_name, combination_type) 
    merged_model = lora_model.merge_and_unload(progressbar=True, adapter_names=['merge']) # merge the final adapter with the model

    # 3. Evaluate on test domain
    print(f"Evaluating final model on test {test_domain} domain")
    test_loss, test_accuracy = evaluate(merged_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return

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
    with open('/leonardo_scratch/fast/IscrC_FoundCL/projects/cl-collab/ModelRatatouille/lquarant/cl-ood/configs/merging_loras.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)



