# import os
# import yaml
# import torch
# import random
# import timm
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
# from data.cl_benchmark_generator import CLBenchmarkGenerator
# import numpy as np
# from collections import defaultdict

# # Utility functions
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

# def require_grad_for_head(model):
#     for name, param in model.named_parameters():
#         if 'head' in name or 'classifier' in name or 'fc' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

# # Data handling
# def prepare_data_loaders(dataset, domains, batch_size):
#     domain_loaders = {}
#     for domain in domains:
#         domain_data = dataset.get_domain_data(domain)
#         train_size = int(0.8 * len(domain_data))
#         val_size = len(domain_data) - train_size
#         train_subset, val_subset = random_split(domain_data, [train_size, val_size])
#         domain_loaders[domain] = {
#             'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True),
#             'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False),
#             'test': DataLoader(domain_data, batch_size=batch_size, shuffle=False)
#         }
#     return domain_loaders

# # Model creation and training
# def create_lora_model(base_model, config):
#     lora_config = LoraConfig(
#         r=config['lora_r'],
#         lora_alpha=config['lora_alpha'],
#         target_modules=["qkv"],
#         lora_dropout=config['lora_dropout'],
#         bias="none"
#     )
#     return get_peft_model(base_model, lora_config)

# def train_epoch(model, data_loader, optimizer, criterion, device):
#     model.train()
#     total_loss, correct, total = 0, 0, 0
#     for inputs, targets, _ in tqdm(data_loader, desc="Training"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#     return total_loss / len(data_loader), 100. * correct / total

# def evaluate(model, data_loader, criterion, device):
#     model.eval()
#     total_loss, correct, total = 0, 0, 0
#     with torch.no_grad():
#         for inputs, targets, _ in tqdm(data_loader, desc="Evaluating"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#     return total_loss / len(data_loader), 100. * correct / total

# def train_lora_adapter(base_model, train_loader, val_loader, test_loader, config, device, domain):
#     print(f"\nTraining LoRA adapter for domain: {domain}")
#     model = create_lora_model(base_model, config).to(device)
#     require_grad_for_head(model)
#     optimizer = optim.AdamW(model.parameters(), lr=float(config['lora_learning_rate']))
#     criterion = nn.CrossEntropyLoss()

#     best_val_accuracy = 0
#     best_model_state = None

#     for epoch in range(config['lora_epochs']):
#         train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
#         val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
#         print(f"Epoch {epoch+1}/{config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#         print(f"Epoch {epoch+1}/{config['lora_epochs']}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             best_model_state = get_peft_model_state_dict(model)

#     model.load_state_dict(best_model_state,strict=False)
#     test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
#     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

#     return best_model_state, test_accuracy

# def train_coefficients(base_model, lora_adapters, train_loader, config, device):
#     coefficients = nn.Parameter(torch.ones(len(lora_adapters), device=device))
#     coefficients.data += torch.randn_like(coefficients) * 0.01
#     torch.nn.utils.clip_grad_value_(coefficients, 1e8)

#     optimizer = optim.Adam([coefficients], lr=1e-2)
#     criterion = nn.CrossEntropyLoss()

#     lora_models = {name: create_lora_model(base_model, config).to(device) for name in lora_adapters.keys()}
#     for name, model in lora_models.items():
#         model.load_state_dict(lora_adapters[name], strict=False)

#     print("Initial coefficients:", F.softmax(coefficients, dim=0).detach().cpu().numpy())

#     for epoch in range(config['weighted_adapter_epochs']):
#         total_loss = 0
#         for inputs, targets, _ in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
            
#             outputs = sum(F.softmax(coefficients, dim=0)[i] * model(inputs) for i, model in enumerate(lora_models.values()))
            
#             loss = criterion(outputs, targets)
#             loss.backward()
            
#             torch.nn.utils.clip_grad_value_(coefficients, 1e8)
#             optimizer.step()
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
#         print(f"Coefficients: {F.softmax(coefficients, dim=0).detach().cpu().numpy()}")

#     return dict(zip(lora_adapters.keys(), F.softmax(coefficients, dim=0).detach().cpu().numpy()))

# def create_merged_model(base_model, lora_adapters, coefficients, config):
#     class MergedModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.base_model = base_model
#             self.adapters = nn.ModuleDict({
#                 name: create_lora_model(base_model, config) for name in lora_adapters.keys()
#             })
#             for name, adapter in self.adapters.items():
#                 adapter.load_state_dict(lora_adapters[name], strict=False)
#             self.coefficients = nn.Parameter(torch.tensor([coefficients[name] for name in lora_adapters.keys()]))

#         def forward(self, x):
#             base_output = self.base_model(x)
#             adapter_outputs = [adapter(x) for adapter in self.adapters.values()]
#             weighted_sum = sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), adapter_outputs))
#             return base_output + weighted_sum

#     return MergedModel()

# # Main experiment function
# def run_baseline_4_experiment(dataset_name, dataset_config, global_config, seed):
#     set_seed(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nRunning Baseline 4 experiment for dataset: {dataset_name}, Seed: {seed}")
#     print(f"Using device: {device}")

#     dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
#     domains = list(set(dataset.domains))
#     test_domain = random.choice(domains)
#     train_domains = [d for d in domains if d != test_domain]
    
#     print(f"Test domain: {test_domain}")
#     print(f"Train domains: {train_domains}")

#     domain_loaders = prepare_data_loaders(dataset, domains, global_config['batch_size'])

#     base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes']).to(device)
#     criterion = nn.CrossEntropyLoss()

#     # Baseline experiment
#     print("\nRunning Baseline experiment (train and test on test domain)")
#     _, baseline_accuracy = train_lora_adapter(base_model, domain_loaders[test_domain]['train'],
#                                               domain_loaders[test_domain]['val'],
#                                               domain_loaders[test_domain]['test'],
#                                               global_config, device, test_domain)
#     print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

#     # Train LoRA adapters for each domain
#     base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes']).to(device)
#     lora_adapters = {}
#     domain_accuracies = defaultdict(dict)

#     for train_domain in train_domains:
#         lora_state_dict, in_domain_accuracy = train_lora_adapter(
#             base_model, domain_loaders[train_domain]['train'],
#             domain_loaders[train_domain]['val'],
#             domain_loaders[train_domain]['test'],
#             global_config, device, train_domain
#         )
#         lora_adapters[train_domain] = lora_state_dict
#         domain_accuracies[train_domain]['in_domain'] = in_domain_accuracy
        
#         test_domain_accuracy = evaluate(create_lora_model(base_model, global_config).to(device).load_state_dict(lora_state_dict, strict=False),
#                                         domain_loaders[test_domain]['test'], criterion, device)[1]
#         domain_accuracies[train_domain]['test_domain'] = test_domain_accuracy

#     print("\nDomain-specific accuracies:")
#     for domain, accuracies in domain_accuracies.items():
#         print(f"  {domain}:")
#         print(f"    In-domain accuracy: {accuracies['in_domain']:.2f}%")
#         print(f"    Test domain accuracy: {accuracies['test_domain']:.2f}%")

#     # Train coefficients and create merged model
#     coefficients = train_coefficients(base_model, lora_adapters, domain_loaders[test_domain]['train'], global_config, device)
    
#     print("\nFinal trained coefficients:")
#     for name, coeff in coefficients.items():
#         print(f"  {name}: {coeff:.4f}")

#     final_model = create_merged_model(base_model, lora_adapters, coefficients, global_config).to(device)
#     final_loss, final_accuracy = evaluate(final_model, domain_loaders[test_domain]['test'], criterion, device)
#     print(f"\nMerged model - Test Loss: {final_loss:.4f}, Test Accuracy: {final_accuracy:.2f}%")

#     base_model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes']).to(device)
#     base_loss, base_accuracy = evaluate(base_model, domain_loaders[test_domain]['test'], criterion, device)
#     print(f"Base model - Test Loss: {base_loss:.4f}, Test Accuracy: {base_accuracy:.2f}%")

#     return {
#         'dataset': dataset_name,
#         'test_domain': test_domain,
#         'train_domains': train_domains,
#         'base_accuracy': base_accuracy,
#         'final_accuracy': final_accuracy,
#         'coefficients': coefficients
#     }

# def main(config):
#     num_runs = config.get('num_runs', 5)
#     all_results = []

#     for run in range(num_runs):
#         print(f"\n--- Run {run + 1}/{num_runs} ---")
#         run_results = []
#         for dataset_name, dataset_config in config['datasets'].items():
#             result = run_baseline_4_experiment(dataset_name, dataset_config, config, seed=run)
#             run_results.append(result)
#         all_results.append(run_results)
    
#     print("\nSummary of All Runs:")
#     for dataset_idx, dataset_name in enumerate(config['datasets'].keys()):
#         base_accuracies = [run[dataset_idx]['base_accuracy'] for run in all_results]
#         final_accuracies = [run[dataset_idx]['final_accuracy'] for run in all_results]
        
#         print(f"\nDataset: {dataset_name}")
#         print(f"  Base Model Accuracy: {np.mean(base_accuracies):.2f}% ± {np.std(base_accuracies):.2f}%")
#         print(f"  Final Merged Model Accuracy: {np.mean(final_accuracies):.2f}% ± {np.std(final_accuracies):.2f}%")
        
#         print("  Average Coefficients:")
#         avg_coefficients = defaultdict(list)
#         for run in all_results:
#             for domain, coeff in run[dataset_idx]['coefficients'].items():
#                 avg_coefficients[domain].append(coeff)
#         for domain, coeffs in avg_coefficients.items():
#             print(f"    {domain}: {np.mean(coeffs):.4f} ± {np.std(coeffs):.4f}")

# if __name__ == "__main__":
#     with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/baseline_configs.yaml', 'r') as file:
#         config = yaml.safe_load(file)
#     main(config)

import os
import yaml
import torch
import random
import timm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
from data.cl_benchmark_generator import CLBenchmarkGenerator
import numpy as np
from collections import defaultdict
import copy
class Baseline4Experiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config['seed'])
        self.dataset_name = config['test_dataset']
        self.dataset_config = config['datasets'][self.dataset_name]
        self.base_model = self.create_base_model()
        self.dataset = self.load_dataset()
        self.domains = list(set(self.dataset.domains))
        self.test_domain = random.choice(self.domains)
        self.train_domains = [d for d in self.domains if d != self.test_domain]
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
            num_classes=self.dataset_config['num_classes']
        ).to(self.device)

    def load_dataset(self):
        return CLBenchmarkGenerator(
            self.dataset_config['path'],
            max_samples_per_class=self.config.get('max_samples_per_class')
        )

    def prepare_data_loaders(self):
        domain_loaders = {}
        for domain in self.domains:
            domain_data = self.dataset.get_domain_data(domain)
            if domain == self.test_domain:
                # Split test domain into 10% for coefficient training and 90% for final evaluation
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
            bias="none"
        )
        model = get_peft_model(base_model, lora_config)
        for name, param in model.named_parameters():
            if 'head' in name or 'classifier' in name or 'fc' in name:
                param.requires_grad = True
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
        model = self.create_lora_model(self.base_model).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=float(self.config['lora_learning_rate']))

        best_val_accuracy = 0
        best_model_state = None

        for epoch in range(self.config['lora_epochs']):
            train_loss, train_accuracy = self.train_epoch(model, self.domain_loaders[domain]['train'], optimizer)
            val_loss, val_accuracy = self.evaluate(model, self.domain_loaders[domain]['val'])
            
            print(f"Epoch {epoch+1}/{self.config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = get_peft_model_state_dict(model)

        model.load_state_dict(best_model_state, strict=False)
        # We'll use the validation set for the final evaluation of the adapter
        _, in_domain_accuracy = self.evaluate(model, self.domain_loaders[domain]['val'])
        print(f"In-domain Accuracy: {in_domain_accuracy:.2f}%")

        return best_model_state, in_domain_accuracy

    def train_coefficients(self):
        print("\nTraining coefficients for adapter merging")
        coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device))
        coefficients.data += torch.randn_like(coefficients) * 0.05
        optimizer = optim.Adam([coefficients], lr=float(self.config['coefficient_lr']))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        lora_models = {name: self.create_lora_model(self.base_model).to(self.device) for name in self.lora_adapters.keys()}
        for name, model in lora_models.items():
            model.load_state_dict(self.lora_adapters[name], strict=False)

        best_accuracy = 0
        best_coefficients = None

        for epoch in range(self.config['coefficient_epochs']):
            total_loss = 0
            correct = 0
            total = 0
            for inputs, targets, _ in self.domain_loaders[self.test_domain]['coeff']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = sum(F.softmax(coefficients, dim=0)[i] * model(inputs) for i, model in enumerate(lora_models.values()))
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_value_(coefficients, 1e8)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.domain_loaders[self.test_domain]['coeff']):.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Coefficients: {F.softmax(coefficients, dim=0).detach().cpu().numpy()}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_coefficients = F.softmax(coefficients, dim=0).detach().clone()
            
            scheduler.step()

        return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))

    def evaluate_all_models(self):
        eval_loader = self.domain_loaders[self.test_domain]['eval']
        
        # Evaluate base model
        base_loss, base_accuracy = self.evaluate(self.base_model, eval_loader)
        print(f"Base model - Test Loss: {base_loss:.4f}, Test Accuracy: {base_accuracy:.2f}%")
        
        # Evaluate individual LoRA adapters
        for domain, adapter_state in self.lora_adapters.items():
            lora_model = self.create_lora_model(self.base_model).to(self.device)
            lora_model.load_state_dict(adapter_state, strict=False)
            loss, accuracy = self.evaluate(lora_model, eval_loader)
            print(f"{domain} adapter - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        # Evaluate merged model
        merged_model = self.create_merged_model(self.coefficients)
        merged_loss, merged_accuracy = self.evaluate(merged_model, eval_loader)
        print(f"Merged model - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")
    
        return base_accuracy, merged_accuracy
    def run_baseline_experiment(self):
        print("\nRunning Baseline experiment (train and test on test domain)")
        # Use a copy of the base model for the baseline experiment
        baseline_model = copy.deepcopy(self.base_model)
        optimizer = optim.AdamW(baseline_model.parameters(), lr=float(self.config['learning_rate']))
        
        best_val_accuracy = 0
        best_model_state = None

        for epoch in range(self.config['num_epochs']):
            train_loss, train_accuracy = self.train_epoch(baseline_model, self.domain_loaders[self.test_domain]['coeff'], optimizer)
            val_loss, val_accuracy = self.evaluate(baseline_model, self.domain_loaders[self.test_domain]['eval'])
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = copy.deepcopy(baseline_model.state_dict())

        baseline_model.load_state_dict(best_model_state)
        _, baseline_accuracy = self.evaluate(baseline_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
        
        return baseline_accuracy

    def create_merged_model(self, coefficients):
        class MergedModel(nn.Module):
            def __init__(self, base_model, lora_adapters, coefficients, config):
                super().__init__()
                self.base_model = base_model
                self.lora_adapters = nn.ModuleDict()
                for name, state_dict in lora_adapters.items():
                    lora_model = get_peft_model(base_model, LoraConfig(
                        r=config['lora_r'],
                        lora_alpha=config['lora_alpha'],
                        target_modules=["qkv"],
                        lora_dropout=config['lora_dropout'],
                        bias="none",
                        task_type=TaskType.FEATURE_EXTRACTION
                    ))
                    lora_model.load_state_dict(state_dict)
                    self.lora_adapters[name] = lora_model
                self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

            def forward(self, x):
                return sum(F.softmax(self.coefficients, dim=0)[i] * adapter(x) 
                        for i, adapter in enumerate(self.lora_adapters.values()))

        return MergedModel(self.base_model, self.lora_adapters, coefficients, self.config).to(self.device)

        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["qkv"],
            lora_dropout=self.config['lora_dropout'],
            bias="none"
        )

        return MergedModel(self.base_model, self.lora_adapters, coefficients, lora_config).to(self.device)

    def run_experiment(self):
        print(f"\nRunning LoRA with Separate Classifiers experiment on {self.config['test_dataset']}")
        print(f"Test domain: {self.test_domain}")
        print(f"Train domains: {self.train_domains}")

        # Train LoRA adapters for each domain
        for train_domain in self.train_domains:
            lora_state, accuracy = self.train_lora_adapter(train_domain)
            self.lora_adapters[train_domain] = lora_state
            print(f"{train_domain} adapter trained. Validation Accuracy: {accuracy:.2f}%")

        # Train coefficients
        coefficients = self.train_coefficients()

        # Create and evaluate merged model
        merged_model = self.create_merged_model(coefficients)
        merged_loss, merged_accuracy = self.evaluate(merged_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"\nMerged model - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")

        # Evaluate base model for comparison
        base_loss, base_accuracy = self.evaluate(self.base_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"Base model - Test Loss: {base_loss:.4f}, Test Accuracy: {base_accuracy:.2f}%")

        return {
            'dataset': self.config['test_dataset'],
            'test_domain': self.test_domain,
            'train_domains': self.train_domains,
            'base_accuracy': base_accuracy,
            'merged_accuracy': merged_accuracy,
            'coefficients': coefficients
        }

def main(config):
    num_runs = config.get('num_runs', 1)
    all_results = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        experiment = Baseline4Experiment(config)
        result = experiment.run_experiment()
        all_results.append(result)
    
    print(f"\nSummary of All Runs for {config['test_dataset']}:")
    base_accuracies = [run['base_accuracy'] for run in all_results]
    final_accuracies = [run['final_accuracy'] for run in all_results]
    
    print(f"Base Model Accuracy: {np.mean(base_accuracies):.2f}% ± {np.std(base_accuracies):.2f}%")
    print(f"Final Merged Model Accuracy: {np.mean(final_accuracies):.2f}% ± {np.std(final_accuracies):.2f}%")
    
    print("Average Coefficients:")
    avg_coefficients = defaultdict(list)
    for run in all_results:
        for domain, coeff in run['coefficients'].items():
            avg_coefficients[domain].append(coeff)
    for domain, coeffs in avg_coefficients.items():
        print(f"  {domain}: {np.mean(coeffs):.4f} ± {np.std(coeffs):.4f}")

if __name__ == "__main__":
    with open('/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/configs/baseline_configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)