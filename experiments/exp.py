import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import random
from copy import deepcopy
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
from data.cl_benchmark_generator import CLBenchmarkGenerator
import timm
from collections import defaultdict
from typing import List, Literal

class TIESLoRAExperiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.load_dataset()
        self.base_model = self.create_base_model()
        self.domain_adapters = {}  # Store LoRA adapters for each domain
        self.test_domain = None
        self.train_domains = None
        self.domain_loaders = None
        self.test_domain_index = 0

    def load_dataset(self):
        return CLBenchmarkGenerator(
            self.config['dataset_path'],
            max_samples_per_class=self.config.get('max_samples_per_class')
        )

    def create_base_model(self):
        return timm.create_model(self.config['base_model'], pretrained=True, num_classes=self.dataset.num_classes)

    def update_head(self, model):
        for name, param in model.named_parameters():
            if 'head' in name:
                param.requires_grad = True
        return model

    def prepare_data(self):
        self.domains = sorted(list(set(self.dataset.domains)))
        num_domains = len(self.domains)
        self.test_domain = self.domains[self.test_domain_index]
        self.test_domain_index = (self.test_domain_index + 1) % num_domains
        self.train_domains = [d for d in self.domains if d != self.test_domain]
        self.domain_loaders = self._create_domain_loaders()
        print(f"Domains: {self.domains}")
        print(f"Selected test domain: {self.test_domain}")
        print(f"Training domains: {self.train_domains}")
        self.dataset.print_samples_per_class_per_domain()

    def _create_domain_loaders(self):
        domain_loaders = {}
        for domain in self.domains:
            domain_data = self.dataset.get_domain_data(domain)
            if domain == self.test_domain:
                train_size = int(0.1 * len(domain_data))
                test_size = len(domain_data) - train_size
                train_subset, test_subset = random_split(domain_data, [train_size, test_size])
                domain_loaders[domain] = {
                    'coeff': DataLoader(train_subset, batch_size=self.config['batch_size'], shuffle=True),
                    'eval': DataLoader(test_subset, batch_size=self.config['batch_size'], shuffle=False)
                }
            else:
                domain_loaders[domain] = {
                    'train': DataLoader(domain_data, batch_size=self.config['batch_size'], shuffle=True)
                }
        return domain_loaders

    def train_and_save_domain_adapter(self, domain):
        print(f"Training model for domain: {domain}")
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
        )
        model = get_peft_model(self.create_base_model(), lora_config)
        model = self.update_head(model)
        model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config['num_epochs']):
            model.train()
            total_loss = 0
            for inputs, labels, _ in tqdm(self.domain_loaders[domain]['train'], desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.domain_loaders[domain]['train'])
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Save the LoRA adapter state
        adapter_weights = get_peft_model_state_dict(model)
        self.domain_adapters[domain] = adapter_weights
        
        return model.cpu()


    def train_coefficients(self):
        print("Training coefficients for weight averaging")
        coefficients = nn.Parameter(torch.ones(len(self.train_domains), device=self.device))
        optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
        criterion = nn.CrossEntropyLoss()

        print(f"Device: {self.device}")
        print(f"Coefficients device: {coefficients.device}")

        # Create LoRA models for each domain
        domain_models = {}
        for domain in self.train_domains:
            base_model = self.create_base_model()
            lora_config = LoraConfig(
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=["qkv", "fc1", "fc2"],
                lora_dropout=self.config['lora_dropout'],
                bias="none",
            )
            model = get_peft_model(base_model, lora_config)
            set_peft_model_state_dict(model, self.domain_adapters[domain])
            domain_models[domain] = model.to(self.device)

        for epoch in range(self.config['coefficient_epochs']):
            total_loss = 0
            for batch_idx, (inputs, labels, _) in enumerate(self.domain_loaders[self.test_domain]['coeff']):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = torch.zeros(inputs.size(0), self.base_model.num_classes, device=self.device)
                for i, domain in enumerate(self.train_domains):
                    domain_models[domain].eval()
                    with torch.no_grad():
                        domain_output = domain_models[domain](inputs)
                    outputs += coefficients[i] * domain_output
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.domain_loaders[self.test_domain]['coeff'])
            print(f"Coefficient Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        return torch.softmax(coefficients, dim=0)

    def create_weight_averaged_model(self, coefficients):
        print("Creating weight-averaged model using TIES merging")
        base_model = self.create_base_model()
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
        )
        avg_model = get_peft_model(base_model, lora_config)
        avg_model.to(self.device)

        merged_adapter = {}
        for key in self.domain_adapters[self.train_domains[0]].keys():
            task_tensors = [self.domain_adapters[domain][key].to(self.device) for domain in self.train_domains]
            weights = coefficients.to(self.device)
            merged_param = ties_merge(task_tensors, weights, density=0.5)
            merged_adapter[key] = merged_param

        # Load the merged adapter weights into the model
        set_peft_model_state_dict(avg_model, merged_adapter)
        
        return avg_model

    def evaluate(self, model, data_loader):
        model = model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def run_experiment(self):
        print("Preparing data...")
        self.prepare_data()
        print(f"Test domain: {self.test_domain}")
        print(f"Training domains: {self.train_domains}")

        print("\nTraining domain adapters...")
        domain_accuracies = {}
        for domain in self.train_domains:
            print(f"\nTraining adapter for domain: {domain}")
            # Change this line from:
            # domain_model = self.train_and_merge_domain_model(domain)
            # to:
            domain_model = self.train_and_save_domain_adapter(domain)
            
            domain_accuracy = self.evaluate(domain_model, self.domain_loaders[self.test_domain]['eval'])
            domain_accuracies[domain] = domain_accuracy
            print(f"Accuracy of {domain} model on test domain: {domain_accuracy:.4f}")

        print("\nTraining coefficients...")
        coefficients = self.train_coefficients()
        print("Coefficient training complete.")
        print("Learned coefficients:", coefficients.tolist())

        print("\nCreating weight-averaged model using TIES merging...")
        final_model = self.create_weight_averaged_model(coefficients)
        print("TIES merged model created.")

        print("\nEvaluating final model...")
        final_accuracy = self.evaluate(final_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"Final model accuracy on test domain: {final_accuracy:.4f}")

        return {
            'test_domain': self.test_domain,
            'domain_accuracies': domain_accuracies,
            'final_accuracy': final_accuracy,
            'coefficients': coefficients.tolist()
        }
# TIES merging functions
def ties_merge(task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float, majority_sign_method: str = "total") -> torch.Tensor:
    task_tensors = [prune(tensor, density, method="magnitude") for tensor in task_tensors]
    task_tensors = torch.stack(task_tensors, dim=0)
    
    majority_sign_mask = calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
    
    weights = reshape_weight_task_tensors(task_tensors, weights)
    weighted_task_tensors = task_tensors * weights
    
    mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
    return mixed_task_tensors

def calculate_majority_sign_mask(tensor: torch.Tensor, method: str = "total") -> torch.Tensor:
    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise ValueError(f"Unknown method {method}")
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign

def disjoint_merge(task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)

def prune(tensor: torch.Tensor, density: float, method: str) -> torch.Tensor:
    if method == "magnitude":
        k = int(density * tensor.numel())
        topk_values, _ = torch.topk(tensor.abs().flatten(), k, largest=True)
        threshold = topk_values[-1]
        return torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))
    else:
        raise ValueError(f"Unknown pruning method {method}")

def reshape_weight_task_tensors(task_tensors, weights):
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


import json
from datetime import datetime

def run_experiments_multiple_datasets(datasets_config, num_runs):
    all_results = {}
    
    for dataset_name, dataset_config in datasets_config.items():
        print(f"\n{'='*50}")
        print(f"Running experiments for dataset: {dataset_name}")
        print(f"{'='*50}")
        
        print(f"Dataset config for {dataset_name}:")
        print(dataset_config)
        
        try:
            dataset_results = run_multiple_experiments(dataset_config, num_runs)
            all_results[dataset_name] = dataset_results
            
            print(f"\nSummary for {dataset_name}:")
            summarize_results(dataset_results)
        except Exception as e:
            print(f"Error running experiments for {dataset_name}: {str(e)}")
    
    return all_results

def run_multiple_experiments(dataset_config, num_runs):
    dataset_results = []
    TIESLoRAExperiment.test_domain_index = 0
    
    try:
        base_seed = int(dataset_config.get('seed', 42))
    except (ValueError, AttributeError) as e:
        print(f"Error getting seed from dataset_config: {str(e)}")
        base_seed = 42
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        config = dataset_config.copy() if isinstance(dataset_config, dict) else {}
        config['seed'] = base_seed + run
        
        try:
            experiment = TIESLoRAExperiment(config)
            results = experiment.run_experiment()
            dataset_results.append(results)
        except Exception as e:
            print(f"Error in run {run + 1}: {str(e)}")
    
    return dataset_results

def summarize_results(results):
    summary = {
        'final_accuracies': [],
        'domain_accuracies': defaultdict(list),
        'coefficients': []
    }

    for result in results:
        if not isinstance(result, dict):
            print(f"Warning: Unexpected result type: {type(result)}. Skipping.")
            continue

        if 'final_accuracy' in result:
            summary['final_accuracies'].append(result['final_accuracy'])
        if 'domain_accuracies' in result:
            for domain, accuracy in result['domain_accuracies'].items():
                summary['domain_accuracies'][domain].append(accuracy)
        if 'coefficients' in result:
            summary['coefficients'].append(result['coefficients'])

    if not summary['final_accuracies']:
        print("No valid results to summarize.")
        return None

    print(f"Final Model Accuracy: {np.mean(summary['final_accuracies']):.4f} ± {np.std(summary['final_accuracies']):.4f}")
    print("Domain-specific Accuracies:")
    for domain, accuracies in summary['domain_accuracies'].items():
        print(f"  {domain}: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

    if summary['coefficients']:
        avg_coefficients = np.mean(summary['coefficients'], axis=0)
        std_coefficients = np.std(summary['coefficients'], axis=0)
        print("Average Coefficients:")
        for i, (avg, std) in enumerate(zip(avg_coefficients, std_coefficients)):
            print(f"  Coefficient {i+1}: {avg:.4f} ± {std:.4f}")

    return summary

if __name__ == "__main__":
    datasets_config = { 
        'PACS': {
            'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/PACS',
            'max_samples_per_class': None,
            'base_model': 'vit_base_patch16_224',
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-3,
            'lora_r': 8,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'coefficient_learning_rate': 1e-1,
            'coefficient_epochs': 10,
            'seed': 42
        },
        # 'VLCS': {
        #     'dataset_path': '/path/to/VLCS/dataset',
        #     'max_samples_per_class': None,
        #     'base_model': 'vit_base_patch16_224',
        #     'batch_size': 32,
        #     'num_epochs': 10,
        #     'learning_rate': 1e-3,
        #     'lora_r': 8,
        #     'lora_alpha': 32,
        #     'lora_dropout': 0.1,
        #     'coefficient_learning_rate': 1e-1,
        #     'coefficient_epochs': 10,
        #     'seed': 42
        # },
        # 'OfficeHome': {
        #     'dataset_path': '/path/to/OfficeHome/dataset',
        #     'max_samples_per_class': None,
        #     'base_model': 'vit_base_patch16_224',
        #     'batch_size': 32,
        #     'num_epochs': 10,
        #     'learning_rate': 1e-3,
        #     'lora_r': 8,
        #     'lora_alpha': 32,
        #     'lora_dropout': 0.1,
        #     'coefficient_learning_rate': 1e-1,
        #     'coefficient_epochs': 10,
        #     'seed': 42
        # }
    }
    
    num_runs = 5  # Number of times to repeat the experiment for each dataset
    
    all_results = run_experiments_multiple_datasets(datasets_config, num_runs)
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"ties_lora_results_{timestamp}.json"
    
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_filename}")
    
    print("\nOverall Summary:")
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        summarize_results(results)