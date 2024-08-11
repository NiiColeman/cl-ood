import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import timm
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm
import random
import numpy as np
from copy import deepcopy
from data.cl_benchmark_generator import CLBenchmarkGenerator
from typing import List, Literal
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR

class LoraTIESExperiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config['seed'])
        self.dataset = CLBenchmarkGenerator(config['dataset_path'], max_samples_per_class=config.get('max_samples_per_class'))
        self.domains = list(set(self.dataset.domains))
        self.test_domain = random.choice(self.domains)
        self.train_domains = [d for d in self.domains if d != self.test_domain]
        self.base_model = self.create_base_model()
        self.domain_loaders = self.prepare_data_loaders()
        self.lora_adapters = {}
        self.criterion = nn.CrossEntropyLoss()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def create_base_model(self):
        return timm.create_model(
            self.config['base_model'],
            pretrained=True,
            num_classes=self.dataset.num_classes
        ).to(self.device)

    def prepare_data_loaders(self):
        domain_loaders = {}
        for domain in self.domains:
            domain_data = self.dataset.get_domain_data(domain)
            if domain == self.test_domain:
                coeff_size = int(0.1 * len(domain_data))
                eval_size = len(domain_data) - coeff_size
                coeff_subset, eval_subset = random_split(domain_data, [coeff_size, eval_size])
                domain_loaders[domain] = {
                    'coeff': DataLoader(coeff_subset, batch_size=self.config['batch_size'], shuffle=True),
                    'eval': DataLoader(eval_subset, batch_size=self.config['batch_size'], shuffle=False)
                }
            else:
                train_size = int(0.8 * len(domain_data))
                val_size = len(domain_data) - train_size
                train_subset, val_subset = random_split(domain_data, [train_size, val_size])
                domain_loaders[domain] = {
                    'train': DataLoader(train_subset, batch_size=self.config['batch_size'], shuffle=True),
                    'val': DataLoader(val_subset, batch_size=self.config['batch_size'], shuffle=False)
                }
        return domain_loaders

    def create_lora_model(self, base_model):
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["qkv"],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        print(f"Created LoRA model: {type(model)}")  # Debug print
        return model

    def train_epoch(self, model, data_loader, optimizer):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets, _ in tqdm(data_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return total_loss / len(data_loader), 100. * correct / total

    def evaluate(self, model, data_loader):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets, _ in tqdm(data_loader, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return total_loss / len(data_loader), 100. * correct / total

    def train_lora_adapter(self, domain):
        print(f"\nTraining LoRA adapter for domain: {domain}")
        model = self.create_lora_model(self.base_model)
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lora_learning_rate'])

        best_val_accuracy = 0
        best_lora_state = None

        for epoch in range(self.config['lora_epochs']):
            train_loss, train_accuracy = self.train_epoch(model, self.domain_loaders[domain]['train'], optimizer)
            val_loss, val_accuracy = self.evaluate(model, self.domain_loaders[domain]['val'])
            
            print(f"Epoch {epoch+1}/{self.config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_lora_state = get_peft_model_state_dict(model)

        if best_lora_state is None:
            print(f"Warning: No best state found for domain {domain}. Using the final state.")
            best_lora_state = get_peft_model_state_dict(model)

        print(f"Best validation accuracy for {domain}: {best_val_accuracy:.2f}%")
        print(f"LoRA state keys: {best_lora_state.keys()}")  # Debug print
        return best_lora_state, best_val_accuracy
    def train_coefficients(self):
        print("\nTraining coefficients for adapter merging")
        coefficients = nn.Parameter(torch.randn(len(self.lora_adapters), device=self.device))
        optimizer = optim.Adam([coefficients], lr=0.01)
        criterion = nn.CrossEntropyLoss()

        lora_models = {}
        for name, lora_state in self.lora_adapters.items():
            lora_model = self.create_lora_model(self.base_model)
            set_peft_model_state_dict(lora_model, lora_state)
            lora_model = lora_model.to(self.device)
            lora_model.eval()
            lora_models[name] = lora_model

        best_accuracy = 0
        best_coefficients = F.softmax(coefficients.detach().clone(), dim=0)

        for epoch in range(self.config['coefficient_epochs']):
            total_loss = 0
            correct = 0
            total = 0
            for inputs, labels, _ in self.domain_loaders[self.test_domain]['coeff']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # Get outputs from all LoRA models
                all_outputs = torch.stack([model(inputs) for model in lora_models.values()])

                # Combine outputs using softmax of coefficients
                soft_coefficients = F.softmax(coefficients, dim=0)
                combined_output = (soft_coefficients.unsqueeze(1).unsqueeze(2) * all_outputs).sum(dim=0)

                loss = criterion(combined_output, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = combined_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.domain_loaders[self.test_domain]['coeff']):.4f}, Accuracy: {accuracy:.2f}%")
            current_coefficients = F.softmax(coefficients, dim=0)
            print(f"Coefficients: {current_coefficients.detach().cpu().numpy()}")
            print(f"Gradients: {coefficients.grad}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_coefficients = current_coefficients.detach().clone()

        print(f"Best coefficient accuracy: {best_accuracy:.2f}%")
        print(f"Final best coefficients: {best_coefficients.cpu().numpy()}")
        return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))



    # def train_coefficients(self):
    #     print("\nTraining coefficients for adapter merging")
    #     coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device))
    #     optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
    #     noise_magnitude = 0.05
        
    #     lora_models = {}
    #     for name, lora_state in self.lora_adapters.items():
    #         lora_model = self.create_lora_model(self.base_model)
    #         set_peft_model_state_dict(lora_model, lora_state)
    #         lora_model = lora_model.to(self.device)
    #         lora_model.eval()
    #         lora_models[name] = lora_model
        
    #     best_accuracy = 0
    #     best_coefficients = F.softmax(coefficients.detach().clone(), dim=0)

    #     for epoch in range(self.config['coefficient_epochs']):
    #         total_loss = 0
    #         correct = 0
    #         total = 0
    #         for inputs, targets, _ in self.domain_loaders[self.test_domain]['coeff']:
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             optimizer.zero_grad()
                
    #             model_outputs = torch.stack([model(inputs) for model in lora_models.values()])
                
    #             soft_coefficients = F.softmax(coefficients, dim=0)
    #             weighted_outputs = (soft_coefficients.unsqueeze(1).unsqueeze(2) * model_outputs).sum(dim=0)
                
    #             loss = self.criterion(weighted_outputs, targets)
    #             loss.backward()
    #             noise = torch.randn_like(coefficients) * noise_magnitude
    #             coefficients.data.add_(noise)
                
    #             print(f"Coefficient gradients: {coefficients.grad}")
                
    #             optimizer.step()
                
    #             total_loss += loss.item()
    #             _, predicted = weighted_outputs.max(1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets).sum().item()
            
    #         accuracy = 100. * correct / total
    #         print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.domain_loaders[self.test_domain]['coeff']):.4f}, Accuracy: {accuracy:.2f}%")
    #         current_coefficients = F.softmax(coefficients, dim=0)
    #         print(f"Coefficients: {current_coefficients.detach().cpu().numpy()}")
            
    #         if accuracy > best_accuracy:
    #             best_accuracy = accuracy
    #             best_coefficients = current_coefficients.detach().clone()

    #     print(f"Best coefficient accuracy: {best_accuracy:.2f}%")
    #     print(f"Final best coefficients: {best_coefficients.cpu().numpy()}")
    #     return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))

    def ties_merge(self, task_tensors: List[torch.Tensor], weights: torch.Tensor, density: float, majority_sign_method: Literal["total", "frequency"] = "total") -> torch.Tensor:
        task_tensors = [self.prune(tensor, density, method="magnitude") for tensor in task_tensors]
        task_tensors = torch.stack(task_tensors, dim=0)
        majority_sign_mask = self.calculate_majority_sign_mask(task_tensors, method=majority_sign_method)
        weights = weights.view(weights.shape + (1,) * (task_tensors.dim() - weights.dim()))
        weighted_task_tensors = task_tensors * weights
        return self.disjoint_merge(weighted_task_tensors, majority_sign_mask)

    def prune(self, tensor: torch.Tensor, density: float, method: Literal["magnitude", "random"]) -> torch.Tensor:
        if method == "magnitude":
            k = int(density * tensor.numel())
            topk_values, _ = torch.topk(tensor.abs().flatten(), k, largest=True)
            threshold = topk_values[-1]
            return torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))
        elif method == "random":
            mask = torch.bernoulli(torch.full_like(tensor, density))
            return tensor * mask
        else:
            raise ValueError(f"Unknown pruning method: {method}")

    def calculate_majority_sign_mask(self, tensor: torch.Tensor, method: Literal["total", "frequency"] = "total") -> torch.Tensor:
        sign = tensor.sign()
        if method == "total":
            sign_magnitude = tensor.sum(dim=0)
        elif method == "frequency":
            sign_magnitude = sign.sum(dim=0)
        else:
            raise ValueError(f"Unknown majority sign method: {method}")
        majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
        return sign == majority_sign

    def disjoint_merge(self, task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
        mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
        num_params_preserved = majority_sign_mask.sum(dim=0)
        return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)

    def create_merged_model_with_ties(self, coefficients):
        class MergedModelWithTIES(nn.Module):
            def __init__(self, base_model, lora_adapters, coefficients):
                super().__init__()
                self.config = config
                self.base_model = base_model
                self.lora_adapters = nn.ModuleDict()
                self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

                for name, lora_state in lora_adapters.items():
                    lora_model = get_peft_model(base_model, LoraConfig(
                        r=self.config['lora_r'],
                        lora_alpha=self.config['lora_alpha'],
                        target_modules=["qkv"],
                        lora_dropout=self.config['lora_dropout'],
                        bias="none",
                    ))
                    set_peft_model_state_dict(lora_model, lora_state)
                    self.lora_adapters[name] = lora_model

            def forward(self, x):
                all_outputs = torch.stack([adapter(x) for adapter in self.lora_adapters.values()])
                soft_coefficients = F.softmax(self.coefficients, dim=0)
                return (soft_coefficients.unsqueeze(1).unsqueeze(2) * all_outputs).sum(dim=0)

        return MergedModelWithTIES(self.base_model, self.lora_adapters, coefficients).to(self.device)

    def run_experiment(self):
        print(f"\nRunning LoRA with TIES experiment")
        print(f"Test domain: {self.test_domain}")
        print(f"Train domains: {self.train_domains}")

        # Baseline experiment
        baseline_model = deepcopy(self.base_model).to(self.device)
        baseline_loss, baseline_accuracy = self.evaluate(baseline_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"Baseline Model - Test Loss: {baseline_loss:.4f}, Test Accuracy: {baseline_accuracy:.2f}%")

        # Train LoRA adapters for each domain
        for train_domain in self.train_domains:
            print(f"\nTraining LoRA adapter for domain: {train_domain}")
            lora_state, accuracy = self.train_lora_adapter(train_domain)
            self.lora_adapters[train_domain] = lora_state
            print(f"{train_domain} adapter trained. Validation Accuracy: {accuracy:.2f}%")

        # Evaluate individual LoRA adapters on the test domain
        print("\nEvaluating individual LoRA adapters on the test domain:")
        individual_accuracies = {}
        for domain, lora_state in self.lora_adapters.items():
            lora_model = self.create_lora_model(self.base_model)
            set_peft_model_state_dict(lora_model, lora_state)
            lora_model = lora_model.to(self.device)
            test_loss, test_accuracy = self.evaluate(lora_model, self.domain_loaders[self.test_domain]['eval'])
            print(f"{domain} adapter - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            individual_accuracies[domain] = test_accuracy

        # Train coefficients
        self.coefficients = self.train_coefficients()

        # Create and evaluate merged model with TIES
        merged_model = self.create_merged_model_with_ties(self.coefficients)
        merged_loss, merged_accuracy = self.evaluate(merged_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"\nMerged model with TIES - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")

        # Debug: Evaluate merged model on a few samples
        print("\nDebugging merged model predictions:")
        merged_model.eval()
        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(self.domain_loaders[self.test_domain]['eval']):
                if i >= 5:  # Only check the first 5 batches
                    break
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = merged_model(inputs)
                _, predicted = outputs.max(1)
                print(f"Batch {i+1}:")
                print(f"  Labels: {labels.cpu().numpy()}")
                print(f"  Predictions: {predicted.cpu().numpy()}")
                print(f"  Accuracy: {100 * (predicted == labels).float().mean().item():.2f}%")

        return {
            'test_domain': self.test_domain,
            'train_domains': self.train_domains,
            'baseline_accuracy': baseline_accuracy,
            'individual_accuracies': individual_accuracies,
            'merged_accuracy': merged_accuracy,
            'coefficients': self.coefficients
        }

def main(config):
    num_runs = config.get('num_runs', 1)
    all_results = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        experiment = LoraTIESExperiment(config)
        result = experiment.run_experiment()
        all_results.append(result)
    
    print("\nSummary of All Runs:")
    baseline_accuracies = [run['baseline_accuracy'] for run in all_results]
    merged_accuracies = [run['merged_accuracy'] for run in all_results]
    
    print(f"Baseline Model Accuracy: {np.mean(baseline_accuracies):.2f}% ± {np.std(baseline_accuracies):.2f}%")
    print(f"Merged Model Accuracy: {np.mean(merged_accuracies):.2f}% ± {np.std(merged_accuracies):.2f}%")
    
    print("Average Coefficients:")
    avg_coefficients = defaultdict(list)
    for run in all_results:
        for domain, coeff in run['coefficients'].items():
            avg_coefficients[domain].append(coeff)
    for domain, coeffs in avg_coefficients.items():
        print(f"  {domain}: {np.mean(coeffs):.4f} ± {np.std(coeffs):.4f}")

if __name__ == "__main__":
    config = {
        'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/PACS',
        'max_samples_per_class': 100,
        'base_model': 'vit_base_patch16_224',
        'batch_size': 32,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        'lora_learning_rate': 1e-4,
        'lora_epochs': 10,
        'coefficient_learning_rate': 1,
        'coefficient_epochs': 10,
        'seed': 42,
        'num_runs': 3,
        'learning_rate': 1e-5,
        'baseline_epochs': 10,
        'ties_density': 0.5,
        'ties_majority_sign_method': 'total'
    }
    main(config)
