import yaml
import torch
import random
import timm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
from data.cl_benchmark_generator import CLBenchmarkGenerator
import numpy as np
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def update_head(model):
    for name,param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model

def train_and_evaluate(model, train_loader, test_loader, config, device):
    optimizer = optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
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
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def train_lora_adapter(base_model, train_loader, config, device, domain):
    print(f"\nTraining LoRA adapter for domain: {domain}")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["qkv", "fc1", "fc2"],
        lora_dropout=config['lora_dropout'],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    
    # Replace the head instead of adding a classifier
    model= update_head(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(config['lora_learning_rate']))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['lora_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['lora_epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # Use the full model forward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{config['lora_epochs']}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return get_peft_model_state_dict(model), model.head.state_dict()

def evaluate_lora_model(base_model, adapter_state_dict, classifier_state_dict, test_loader, config, device):
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["qkv", "fc1", "fc2"],
        lora_dropout=config['lora_dropout'],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.load_state_dict(adapter_state_dict, strict=False)
    model.classifier = nn.Linear(base_model.num_features, base_model.num_classes).to(device)
    model.classifier.load_state_dict(classifier_state_dict)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.base_model.forward_features(inputs)
            outputs = model.classifier(features)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def train_coefficients(base_model, lora_adapters, train_loader, config, device):
    print("\nTraining coefficients for weighted average of LoRA adapters and heads")
    coefficients = nn.Parameter(torch.ones(len(lora_adapters), device=device) / len(lora_adapters))
    coefficients.data += torch.randn_like(coefficients) * 0.02  # Slight asymmetry in initialization
    optimizer = optim.Adam([coefficients], lr=float(config['coefficient_lr']))
    criterion = nn.CrossEntropyLoss()

    lora_models = {}
    for name, (adapter_state_dict, head_state_dict) in lora_adapters.items():
        lora_model = get_peft_model(base_model, LoraConfig(
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=config['lora_dropout'],
            bias="none"
        )).to(device)
        lora_model.load_state_dict(adapter_state_dict, strict=False)
        lora_model.head = nn.Linear(base_model.num_features, base_model.num_classes).to(device)
        lora_model.head.load_state_dict(head_state_dict)
        lora_models[name] = lora_model

    print("Initial coefficients:", coefficients.detach().cpu().numpy())

    for epoch in range(config['coefficient_epochs']):
        epoch_loss = 0.0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Coefficient Epoch {epoch+1}/{config['coefficient_epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = sum(F.softmax(coefficients, dim=0)[i] * model(inputs)
                          for i, (_, model) in enumerate(lora_models.items()))
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
        print(f"Coefficients after epoch {epoch+1}:", coefficients.detach().cpu().numpy())

    final_coefficients = coefficients.detach().cpu().numpy()
    print("Final coefficients:")
    for name, coeff in zip(lora_adapters.keys(), final_coefficients):
        print(f"  {name}: {coeff:.4f}")

    return dict(zip(lora_adapters.keys(), final_coefficients))
    
def create_merged_model(base_model, lora_adapters, coefficients, config):
    class MergedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model
            self.adapters = nn.ModuleDict()
            self.heads = nn.ModuleDict()
            for name, (adapter_state_dict, head_state_dict) in lora_adapters.items():
                adapter = get_peft_model(base_model, LoraConfig(
                    r=config['lora_r'],
                    lora_alpha=config['lora_alpha'],
                    target_modules=["qkv", "fc1", "fc2"],
                    lora_dropout=config['lora_dropout'],
                    bias="none"
                ))
                adapter.load_state_dict(adapter_state_dict, strict=False)
                self.adapters[name] = adapter
                
                head = nn.Linear(base_model.num_features, base_model.num_classes)
                head.load_state_dict(head_state_dict)
                self.heads[name] = head
            
            self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

        def forward(self, x):
            base_features = self.base_model.forward_features(x)
            adapted_features = sum(
                F.softmax(self.coefficients, dim=0)[i] * adapter.forward_features(x)
                for i, adapter in enumerate(self.adapters.values())
            )
            combined_features = base_features + adapted_features
            
            outputs = sum(
                F.softmax(self.coefficients, dim=0)[i] * head(combined_features)
                for i, head in enumerate(self.heads.values())
            )
            
            return outputs

    return MergedModel()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def run_baseline_4_experiment(dataset_name, dataset_config, global_config, seed):
    set_seed(seed)
    print(f"\nRunning Baseline 4 experiment for dataset: {dataset_name}, Seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
    domains = list(set(dataset.domains))
    test_domain = random.choice(domains)
    train_domains = [d for d in domains if d != test_domain]
    
    print(f"Test domain: {test_domain}")
    print(f"Train domains: {train_domains}")

    test_subset = dataset.get_domain_data(test_domain)
    test_loader = DataLoader(test_subset, batch_size=global_config['batch_size'], shuffle=False)
    
    # Baseline experiment: train and test on test domain
    print("\nRunning Baseline experiment (train and test on test domain)")
    base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes']).to(device)
    train_loader = DataLoader(test_subset, batch_size=global_config['batch_size'], shuffle=True)
    baseline_accuracy = train_and_evaluate(base_model, train_loader, test_loader, global_config, device)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    # Reset model for Baseline 4
    base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes']).to(device)
    
    lora_adapters = {}
    for domain in train_domains:
        train_subset = dataset.get_domain_data(domain)
        train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)
        adapter_state_dict, classifier_state_dict = train_lora_adapter(base_model, train_loader, global_config, device, domain)
        lora_adapters[domain] = (adapter_state_dict, classifier_state_dict)
        
        # Evaluate base model with this domain's adapter and classifier
        accuracy = evaluate_lora_model(base_model, adapter_state_dict, classifier_state_dict, test_loader, global_config, device)
        print(f"Accuracy after training on {domain}: {accuracy:.2f}%")

    coefficients = train_coefficients(base_model, lora_adapters, DataLoader(test_subset, batch_size=global_config['batch_size'], shuffle=True), global_config, device)
    
    print("\nFinal trained coefficients:")
    for name, coeff in coefficients.items():
        print(f"  {name}: {coeff:.4f}")

    final_model = create_merged_model(base_model, lora_adapters, coefficients, global_config).to(device)
    accuracy = evaluate(final_model, test_loader, device)
    print(f"Baseline 4 - Final Test Accuracy: {accuracy:.2f}%")
    
    return {
        'dataset': dataset_name,
        'test_domain': test_domain,
        'train_domains': train_domains,
        'baseline_accuracy': baseline_accuracy,
        'final_accuracy': accuracy,
        'coefficients': coefficients
    }

def main(config):
    num_runs = config.get('num_runs', 5)  # Default to 5 runs if not specified
    all_results = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        run_results = []
        for dataset_name, dataset_config in config['datasets'].items():
            result = run_baseline_4_experiment(dataset_name, dataset_config, config, seed=run)
            run_results.append(result)
        all_results.append(run_results)
    
    print("\nSummary of All Runs:")
    # for dataset_idx, dataset_name in enumerate(config['datasets'].keys()):
    #     baseline_accuracies = [