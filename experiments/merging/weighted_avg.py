import os
import copy
import yaml
import timm
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from data.cl_benchmark_generator import CLBenchmarkGenerator
from torch.utils.data import DataLoader, Subset, random_split
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
import matplotlib.pyplot as plt

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

import copy
import inspect

def get_model_args(model):
    """Extract the arguments used to create the model."""
    # Get the signature of the model's __init__ method
    signature = inspect.signature(type(model).__init__)
    
    # Extract the parameter names
    param_names = list(signature.parameters.keys())
    
    # Remove 'self' if it's present
    if 'self' in param_names:
        param_names.remove('self')
    
    # Create a dictionary of arguments
    args = {}
    for name in param_names:
        if hasattr(model, name):
            value = getattr(model, name)
            # Convert tensor attributes to their scalar values if they're size 1
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            args[name] = value
    
    return args


def weighted_adapter_merge(base_model, lora_adapters, coefficients):
    merged_state_dict = OrderedDict()
    base_state_dict = base_model.state_dict()
    
    for key in base_state_dict.keys():
        merged_state_dict[key] = base_state_dict[key].clone()
        
        for adapter_name, adapter_state_dict in lora_adapters.items():
            if key in adapter_state_dict:
                merged_state_dict[key] += coefficients[adapter_name] * adapter_state_dict[key]
    
    return merged_state_dict

def create_and_load_model(base_model, merged_state_dict):
    new_model = type(base_model)(num_classes=base_model.num_classes)
    
    # Filter out unexpected keys
    filtered_state_dict = {k: v for k, v in merged_state_dict.items() if k in new_model.state_dict()}
    
    # Load the filtered state dict
    new_model.load_state_dict(filtered_state_dict, strict=False)
    
    # Check for missing keys
    missing_keys = set(new_model.state_dict().keys()) - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")
    
    return new_model

def train_adapter_coefficients(base_model, lora_adapters, train_loader, val_loader, device, learning_rate, num_epochs):
    """
    Train coefficients for weighted average of LoRA adapters.

  
    -This code snippet initializes a dictionary called coefficients using a dictionary comprehension. 
    The keys of the dictionary are the names of the lora_adapters and the values are PyTorch nn.Parameter objects.

    -The nn.Parameter function is used to create a tensor that can be optimized during training. 
    In this case, the tensor is initialized with the value 1.0 / len(lora_adapters), which is the reciprocal of the length of the lora_adapters list.

    -By using a dictionary comprehension, the code creates a key-value pair for each name in the lora_adapters list, 
    the key is the name and the value is the nn.Parameter object.

    -The optimizer is initialized using the Adam optimizer with the coefficients as the parameters to be optimized.
    
    """
    coefficients = {name: nn.Parameter(torch.tensor(1.0 / len(lora_adapters))) for name in lora_adapters}
    # coefficients.data += torch.randn_like(coefficients) * 0.01 
    
    optimizer = optim.Adam(coefficients.values(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            merged_state_dict = weighted_adapter_merge(base_model, lora_adapters, coefficients)
            model = create_and_load_model(base_model, merged_state_dict)
            model.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")

    return coefficients

def analyze_coefficients(coefficients, lora_adapters):
    print("\nAnalysis of LoRA Adapter Coefficients:")
    print("---------------------------------------")
    
    # Convert coefficients to a normalized form (if not already)
    coef_values = torch.tensor([coef.item() for coef in coefficients.values()])
    normalized_coef = torch.nn.functional.softmax(coef_values, dim=0)
    
    # Create a dictionary of domain names and their normalized coefficients
    coef_dict = {name: value.item() for name, value in zip(lora_adapters.keys(), normalized_coef)}
    
    # Sort coefficients by value
    sorted_coef = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Print coefficients
    for domain, value in sorted_coef:
        print(f"{domain}: {value:.4f}")
    
    # Calculate and print statistics
    max_domain, max_value = max(coef_dict.items(), key=lambda x: x[1])
    min_domain, min_value = min(coef_dict.items(), key=lambda x: x[1])
    avg_value = sum(coef_dict.values()) / len(coef_dict)
    
    print(f"\nHighest contribution: {max_domain} ({max_value:.4f})")
    print(f"Lowest contribution: {min_domain} ({min_value:.4f})")
    print(f"Average contribution: {avg_value:.4f}")
    
    # Calculate the ratio of highest to lowest
    ratio = max_value / min_value if min_value > 0 else float('inf')
    print(f"Ratio of highest to lowest: {ratio:.2f}")
    
    # Visualize the coefficients
    plt.figure(figsize=(10, 6))
    plt.bar(coef_dict.keys(), coef_dict.values())
    plt.title("LoRA Adapter Coefficients")
    plt.xlabel("Domain")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('lora_coefficients.png')
    plt.close()
    
    print("\nA bar chart of the coefficients has been saved as 'lora_coefficients.png'")

    return coef_dict



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

    # Create base model
    base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
    base_model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Prepare test data
    test_subset = dataset.get_domain_data(test_domain)
    test_loader = DataLoader(test_subset, batch_size=global_config['batch_size'], shuffle=False)

    # Baseline 1: Train and test on test domain only
    print("Baseline 1: Training and testing on test domain only")
    base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])

    model = base_model
    optimizer = optim.AdamW(model.parameters(), lr=float(global_config['learning_rate']))
    model = train(model, test_loader, criterion, optimizer, device, global_config['num_epochs'])
    baseline1_loss, baseline1_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Baseline 1 - Test Loss: {baseline1_loss:.4f}, Test Accuracy: {baseline1_accuracy:.2f}%")

    # # Baseline 2: Train on other domains with LoRA, then test on test domain
    print("Baseline 2: Training on other domains with LoRA, then testing on test domain")
    base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])

    model = base_model
    lora_adapters = {}
    for train_domain in train_domains:
        print(f"Training on domain: {train_domain}")
        train_subset = dataset.get_domain_data(train_domain)
        train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)

        lora_config = LoraConfig(
            r=global_config['lora_r'],
            lora_alpha=global_config['lora_alpha'],
            target_modules=["qkv"],
            lora_dropout=global_config['lora_dropout'],
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model = lora_model.to(device)

        optimizer = optim.AdamW(lora_model.parameters(), lr=float(global_config['learning_rate']))
        lora_model = train(lora_model, train_loader, criterion, optimizer, device, global_config['num_epochs'])

        lora_weights = get_peft_model_state_dict(lora_model)
        lora_adapters[train_domain] = lora_weights

        model = lora_model.merge_and_unload()

    baseline2_loss, baseline2_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Baseline 2 - Test Loss: {baseline2_loss:.4f}, Test Accuracy: {baseline2_accuracy:.2f}%")

    # Baseline 3: Sequential LoRA training with weight merging
    print("Baseline 3: Sequential LoRA training with weight merging")
    model = base_model
    for train_domain in train_domains:
        print(f"Training on domain: {train_domain}")
        train_subset = dataset.get_domain_data(train_domain)
        train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)

        lora_config = LoraConfig(
            r=global_config['lora_r'],
            lora_alpha=global_config['lora_alpha'],
            target_modules=["qkv"],
            lora_dropout=global_config['lora_dropout'],
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model = lora_model.to(device)

        optimizer = optim.AdamW(lora_model.parameters(), lr=float(global_config['learning_rate']))
        lora_model = train(lora_model, train_loader, criterion, optimizer, device, global_config['num_epochs'])

        model = lora_model.merge_and_unload()

    baseline3_loss, baseline3_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Baseline 3 - Test Loss: {baseline3_loss:.4f}, Test Accuracy: {baseline3_accuracy:.2f}%")

    Baseline 4: Weighted average of LoRA adapters
    
    print("Baseline 4: Weighted average of LoRA adapters")
    
    # Prepare a small subset of the test domain for training coefficients
    num_samples = len(test_subset)
    num_train_samples = int(0.1 * num_samples)  # Use 10% of test data for training coefficients
    train_subset, val_subset = random_split(test_subset, [num_train_samples, num_samples - num_train_samples])

    coef_train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)
    coef_val_loader = DataLoader(val_subset, batch_size=global_config['batch_size'], shuffle=False)

    # Train coefficients for weighted average
    print("Training coefficients for weighted average of LoRA adapters")
    coefficients = train_adapter_coefficients(
        base_model, lora_adapters, coef_train_loader, coef_val_loader, device,
        learning_rate=global_config['weighted_adapter_lr'],
        num_epochs=global_config['weighted_adapter_epochs']
    )

    print("Analyzing learned coefficients")
    coef_dict = analyze_coefficients(coefficients, lora_adapters)
    # Create final merged model
    merged_state_dict = weighted_adapter_merge(base_model, lora_adapters, coefficients)
    final_model = create_and_load_model(base_model, merged_state_dict)
    final_model = final_model.to(device)

    # Evaluate on the remaining test data
    baseline4_loss, baseline4_accuracy = evaluate(final_model, DataLoader(val_subset, batch_size=global_config['batch_size'], shuffle=False), criterion, device)
    print(f"Baseline 4 - Test Loss: {baseline4_loss:.4f}, Test Accuracy: {baseline4_accuracy:.2f}%")

    # Save results and models
    results = {
        'dataset': dataset_name,
        'test_domain': test_domain,
        'train_domains': train_domains,
        'baseline1_accuracy': baseline1_accuracy,
        'baseline2_accuracy': baseline2_accuracy,
        # 'baseline3_accuracy': baseline3_accuracy,
        'baseline4_accuracy': baseline4_accuracy
    }

    output_dir = os.path.join(global_config['output_dir'], dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    with open(os.path.join(output_dir, 'results.yaml'), 'w') as f:
        yaml.dump(results, f)

    # Save models
    torch.save(base_model.state_dict(), os.path.join(output_dir, 'base_model.pth'))
    torch.save(lora_adapters, os.path.join(output_dir, 'lora_adapters.pth'))
    torch.save(coefficients, os.path.join(output_dir, 'weighted_coefficients.pth'))
    torch.save(final_model.state_dict(), os.path.join(output_dir, 'final_weighted_model.pth'))

    return results

def main(config):
    results = []
    for dataset_name, dataset_config in config['datasets'].items():
        result = run_baseline_experiments_for_dataset(dataset_name, dataset_config, config)
        results.append(result)
    
    # Print summary of results
    print("\nSummary of Results:")
    for result in results:
        print(f"Dataset: {result['dataset']}")
        print(f"  Test Domain: {result['test_domain']}")
        print(f"  Train Domains: {result['train_domains']}")
        print(f"  Baseline 1 Accuracy: {result['baseline1_accuracy']:.2f}%")
        print(f"  Baseline 2 Accuracy: {result['baseline2_accuracy']:.2f}%")
        # print(f"  Baseline 3 Accuracy: {result['baseline3_accuracy']:.2f}%")
        print(f"  Baseline 4 Accuracy: {result['baseline4_accuracy']:.2f}%")
        print("---")

if __name__ == "__main__":
    with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/baseline_configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)