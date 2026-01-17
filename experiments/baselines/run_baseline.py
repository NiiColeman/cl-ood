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

class Baseline4Experiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config['seed'])
        self.dataset = self.load_dataset()
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

    def load_dataset(self):
        return CLBenchmarkGenerator(
            self.config['dataset_path'],
            max_samples_per_class=self.config.get('max_samples_per_class')
        )

    def create_base_model(self):
        return timm.create_model(
            self.config['base_model'],
            pretrained=True,
            num_classes=self.dataset.num_classes
        ).to(self.device)

    def update_head(self,model):
        for name, param in model.named_parameters():
            if "head" in name:
                param.requires_grad = True
        
        return model
            

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
        return model


    ################################## training ##################################

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
        model = self.update_head(model)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lora_learning_rate'])

        best_val_accuracy = 0
        best_lora_state = None
        best_head_state = None

        for epoch in range(self.config['lora_epochs']):
            train_loss, train_accuracy = self.train_epoch(model, self.domain_loaders[domain]['train'], optimizer)
            val_loss, val_accuracy = self.evaluate(model, self.domain_loaders[domain]['val'])
            
            print(f"Epoch {epoch+1}/{self.config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_lora_state = get_peft_model_state_dict(model)
                best_head_state = deepcopy(model.base_model.head.state_dict())

        return best_lora_state, best_head_state, best_val_accuracy



    ################################## adaptation ##################################
    def train_coefficients(self):
        print("\nTraining coefficients for adapter merging")
        # coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device))

        coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device) / len(self.lora_adapters))

        coefficients.data += torch.randn_like(coefficients) * 0.01
        optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        print(f"Initial coefficients: {F.softmax(coefficients, dim=0).detach().cpu().numpy()}")

        lora_models = {}
        for name, (lora_state, head_state) in self.lora_adapters.items():
            lora_model = self.create_lora_model(self.base_model)
            set_peft_model_state_dict(lora_model, lora_state)
            lora_model.base_model.head.load_state_dict(head_state)
            lora_models[name] = lora_model.to(self.device)

        best_accuracy = 0
        best_coefficients = F.softmax(coefficients, dim=0).detach().clone()

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
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.domain_loaders[self.test_domain]['coeff']):.4f}, Accuracy: {accuracy:.2f}%")
            current_coefficients = F.softmax(coefficients, dim=0)
            print(f"Coefficients: {current_coefficients.detach().cpu().numpy()}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_coefficients = current_coefficients.detach().clone()
            
            # scheduler.step()

        print(f"Best coefficient accuracy: {best_accuracy:.2f}%")
        print(f"Final best coefficients: {best_coefficients.cpu().numpy()}")
        return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))

    def create_merged_model(self, coefficients):
        class MergedModel(nn.Module):
            def __init__(self, base_model, lora_adapters, coefficients, config):
                super().__init__()
                self.base_model = base_model
                self.lora_adapters = nn.ModuleDict()
                for name, (lora_state, head_state) in lora_adapters.items():
                    lora_model = get_peft_model(base_model, LoraConfig(
                        r=config['lora_r'],
                        lora_alpha=config['lora_alpha'],
                        target_modules=["qkv"],
                        lora_dropout=config['lora_dropout'],
                        bias="none",
                    ))
                    set_peft_model_state_dict(lora_model, lora_state)
                    lora_model.base_model.head.load_state_dict(head_state)
                    self.lora_adapters[name] = lora_model
                self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

            def forward(self, x):
                outputs = [adapter(x) for adapter in self.lora_adapters.values()]
                return sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), outputs))

        return MergedModel(self.base_model, self.lora_adapters, coefficients, self.config).to(self.device)


    # def create_merged_model(self, coefficients):
    #     class MergedModel(nn.Module):
    #         def __init__(self, base_model, lora_adapters, coefficients, config):
    #             super().__init__()
    #             self.base_model = base_model
    #             self.lora_adapters = nn.ModuleDict()
    #             self.classifiers = nn.ModuleDict()
    #             self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

    #             for name, (lora_state, head_state) in lora_adapters.items():
    #                 lora_model = get_peft_model(base_model, LoraConfig(
    #                     r=config['lora_r'],
    #                     lora_alpha=config['lora_alpha'],
    #                     target_modules=["qkv"],
    #                     lora_dropout=config['lora_dropout'],
    #                     bias="none",
    #                 ))
    #                 set_peft_model_state_dict(lora_model, lora_state)
    #                 self.lora_adapters[name] = lora_model

    #                 classifier = nn.Linear(base_model.head.in_features, base_model.head.out_features)
    #                 classifier.load_state_dict(head_state)
    #                 self.classifiers[name] = classifier

    #             # Add a new linear layer to match dimensions
    #             self.feature_adapter = nn.Linear(7, base_model.head.in_features)

    #         def forward(self, x):
    #             # Apply LoRA adapters
    #             lora_outputs = [adapter(x) for adapter in self.lora_adapters.values()]
    #             weighted_lora_output = sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), lora_outputs))

    #             # Adapt the features to match the classifier input size
    #             adapted_features = self.feature_adapter(weighted_lora_output)

    #             # Apply classifiers
    #             classifier_outputs = [classifier(adapted_features) for classifier in self.classifiers.values()]
    #             weighted_classifier_output = sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), classifier_outputs))

    #             return weighted_classifier_output

    #     return MergedModel(self.base_model, self.lora_adapters, coefficients, self.config).to(self.device)


    ################################## main ##################################
    def run_baseline_experiment(self):
        print("\nRunning Baseline experiment (train and test on test domain)")
        baseline_model = deepcopy(self.base_model).to(self.device)
        optimizer = optim.AdamW(baseline_model.parameters(), lr=self.config['learning_rate'])
        
        best_val_accuracy = 0
        best_model_state = None

        train_loader = self.domain_loaders[self.test_domain]['coeff']
        val_loader = self.domain_loaders[self.test_domain]['eval']

        for epoch in range(self.config['baseline_epochs']):
            train_loss, train_accuracy = self.train_epoch(baseline_model, train_loader, optimizer)
            val_loss, val_accuracy = self.evaluate(baseline_model, val_loader)
            
            print(f"Epoch {epoch+1}/{self.config['baseline_epochs']}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = deepcopy(baseline_model.state_dict())

        baseline_model.load_state_dict(best_model_state)
        baseline_loss, baseline_accuracy = self.evaluate(baseline_model, val_loader)
        print(f"Baseline Model - Final Test Loss: {baseline_loss:.4f}, Test Accuracy: {baseline_accuracy:.2f}%")
        
        return baseline_accuracy

    # def run_experiment(self):
    #     print(f"\nRunning LoRA with Separate Heads experiment")
    #     print(f"Test domain: {self.test_domain}")
    #     print(f"Train domains: {self.train_domains}")

    #     base_accuracy = self.run_baseline_experiment()
        
    #     # Train LoRA adapters for each domain
    #     for train_domain in self.train_domains:
    #         lora_state, head_state, accuracy = self.train_lora_adapter(train_domain)
    #         self.lora_adapters[train_domain] = (lora_state, head_state)
    #         print(f"{train_domain} adapter trained. Validation Accuracy: {accuracy:.2f}%")

    #     # Train coefficients
    #     self.coefficients = self.train_coefficients()

    #     # Create and evaluate merged model
    #     merged_model = self.create_merged_model(self.coefficients)
    #     merged_loss, merged_accuracy = self.evaluate(merged_model, self.domain_loaders[self.test_domain]['eval'])
    #     print(f"\nMerged model - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")

    #     return {
    #         'test_domain': self.test_domain,
    #         'train_domains': self.train_domains,
    #         'base_accuracy': base_accuracy,
    #         'merged_accuracy': merged_accuracy,
    #         'coefficients': self.coefficients
    # }
    def run_experiment(self):
        print(f"\nRunning LoRA with Separate Classifiers experiment")
        print(f"Test domain: {self.test_domain}")
        print(f"Train domains: {self.train_domains}")

        base_accuracy = self.run_baseline_experiment()
        
        # Train LoRA adapters for each domain
        self.lora_adapters = {}
        for train_domain in self.train_domains:
            lora_state, classifier_state, accuracy = self.train_lora_adapter(train_domain)
            self.lora_adapters[train_domain] = (lora_state, classifier_state)
            print(f"{train_domain} adapter trained. Validation Accuracy: {accuracy:.2f}%")

        # Train coefficients
        self.coefficients = self.train_coefficients()

        # Create and evaluate merged model
        merged_model = self.create_merged_model(self.coefficients)
        merged_loss, merged_accuracy = self.evaluate(merged_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"\nMerged model - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")

        return {
            'test_domain': self.test_domain,
            'train_domains': self.train_domains,
            'base_accuracy': base_accuracy,
            'merged_accuracy': merged_accuracy,
            'coefficients': self.coefficients
        }
import numpy as np
from collections import defaultdict

def main(config):
    num_runs = config.get('num_runs', 1)
    all_results = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        experiment = Baseline4Experiment(config)
        result = experiment.run_experiment()
        all_results.append(result)
    
    print("\nSummary of All Runs:")
    base_accuracies = [run['base_accuracy'] for run in all_results]
    merged_accuracies = [run['merged_accuracy'] for run in all_results]
    
    print(f"Base Model Accuracy: {np.mean(base_accuracies):.2f}% ± {np.std(base_accuracies):.2f}%")
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
        'max_samples_per_class':500,
        'base_model': 'vit_base_patch16_224',
        'batch_size': 32,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_learning_rate': 1e-5,
        'lora_epochs': 10,
        'coefficient_learning_rate': 2,
        'coefficient_epochs': 10,
        'seed': 42,
        'num_runs': 10,
        'learning_rate': 1e-4,
        'baseline_epochs': 10  
    }
    main(config)
   

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split
# import timm
# from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
# from tqdm import tqdm
# import random
# import numpy as np
# from copy import deepcopy
# from collections import defaultdict

# # Assuming you have a CLBenchmarkGenerator class in data.cl_benchmark_generator
# from data.cl_benchmark_generator import CLBenchmarkGenerator

# import random
# from itertools import cycle

# class Baseline4Experiment:
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.set_seed(config['seed'])
#         self.dataset = self.load_dataset()
#         self.domains = sorted(list(set(self.dataset.domains)))
#         self.base_model = self.create_base_model()
#         self.lora_adapters = {}
#         self.criterion = nn.CrossEntropyLoss()
#         self.domain_cycle = self.create_domain_cycle()
#         self.test_domain = None
#         self.train_domains = None
#         self.domain_loaders = None
#         self.test_domain_index = 0

#     def create_domain_cycle(self):
#         domains_copy = self.domains.copy()
#         random.shuffle(domains_copy)
#         return cycle(domains_copy)

#     def prepare_data(self):
#         self.test_domain = next(self.domain_cycle)
#         self.train_domains = [d for d in self.domains if d != self.test_domain]
#         self.domain_loaders = self.prepare_data_loaders()
        
#         print(f"Domains: {self.domains}")
#         print(f"Selected test domain: {self.test_domain}")
#         print(f"Training domains: {self.train_domains}")

#     def prepare_data_loaders(self):
#         domain_loaders = {}
#         for domain in self.domains:
#             domain_data = self.dataset.get_domain_data(domain)
#             if domain == self.test_domain:
#                 coeff_size = int(0.1 * len(domain_data))
#                 eval_size = len(domain_data) - coeff_size
#                 coeff_subset, eval_subset = random_split(domain_data, [coeff_size, eval_size])
#                 domain_loaders[domain] = {
#                     'coeff': DataLoader(coeff_subset, batch_size=self.config['batch_size'], shuffle=True),
#                     'eval': DataLoader(eval_subset, batch_size=self.config['batch_size'], shuffle=False)
#                 }
#             else:
#                 train_size = int(0.8 * len(domain_data))
#                 val_size = len(domain_data) - train_size
#                 train_subset, val_subset = random_split(domain_data, [train_size, val_size])
#                 domain_loaders[domain] = {
#                     'train': DataLoader(train_subset, batch_size=self.config['batch_size'], shuffle=True),
#                     'val': DataLoader(val_subset, batch_size=self.config['batch_size'], shuffle=False)
#                 }
#         return domain_loaders

#     def set_seed(self, seed):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)

#     def load_dataset(self):
#         return CLBenchmarkGenerator(
#             self.config['dataset_path'],
#             max_samples_per_class=self.config.get('max_samples_per_class')
#         )

#     def create_base_model(self):
#         return timm.create_model(
#             self.config['base_model'],
#             pretrained=True,
#             num_classes=self.dataset.num_classes
#         ).to(self.device)


#     def update_head(self,model):
#         for name, param in model.named_parameters():
#             if "head" in name:
#                 param.requires_grad = True
        
#         return model

#     def create_lora_model(self, base_model):
#         lora_config = LoraConfig(
#             r=self.config['lora_r'],
#             lora_alpha=self.config['lora_alpha'],
#             target_modules=["qkv"],
#             lora_dropout=self.config['lora_dropout'],
#             bias="none",
#         )
#         return get_peft_model(base_model, lora_config)

#     def train_epoch(self, model, data_loader, optimizer):
#         model.train()
#         total_loss, correct, total = 0, 0, 0
#         for inputs, targets, _ in tqdm(data_loader, desc="Training", leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = self.criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#         return total_loss / len(data_loader), 100. * correct / total

#     def evaluate(self, model, data_loader):
#         model.eval()
#         total_loss, correct, total = 0, 0, 0
#         with torch.no_grad():
#             for inputs, targets, _ in tqdm(data_loader, desc="Evaluating", leave=False):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 total_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#         return total_loss / len(data_loader), 100. * correct / total

#     def train_lora_adapter(self, domain):
#         print(f"\nTraining LoRA adapter for domain: {domain}")
#         model = self.create_lora_model(self.base_model)
#         optimizer = optim.AdamW(model.parameters(), lr=self.config['lora_learning_rate'])

#         best_val_accuracy = 0
#         best_lora_state = None
#         best_head_state = None

#         for epoch in range(self.config['lora_epochs']):
#             train_loss, train_accuracy = self.train_epoch(model, self.domain_loaders[domain]['train'], optimizer)
#             val_loss, val_accuracy = self.evaluate(model, self.domain_loaders[domain]['val'])
            
#             print(f"Epoch {epoch+1}/{self.config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#             print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
#             if val_accuracy > best_val_accuracy:
#                 best_val_accuracy = val_accuracy
#                 best_lora_state = get_peft_model_state_dict(model)
#                 best_head_state = deepcopy(model.base_model.head.state_dict())

#         if best_lora_state is None:
#             print(f"Warning: No best LoRA state found for domain {domain}. Using the final state.")
#             best_lora_state = get_peft_model_state_dict(model)

#         if best_head_state is None:
#             print(f"Warning: No best head state found for domain {domain}. Using the final state.")
#             best_head_state = deepcopy(model.base_model.head.state_dict())

#         return best_lora_state, best_head_state, best_val_accuracy

#     def train_coefficients(self):
#         print("\nTraining coefficients for adapter merging")
#         coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device) / len(self.lora_adapters))
#         coefficients.data += torch.randn_like(coefficients) * 0.01
#         optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
        
#         print(f"Initial coefficients: {F.softmax(coefficients, dim=0).detach().cpu().numpy()}")

#         lora_models = {}
#         for name, (lora_state, head_state) in self.lora_adapters.items():
#             if lora_state is None or head_state is None:
#                 print(f"Warning: LoRA state or head state is None for domain {name}. Skipping this adapter.")
#                 continue

#             lora_model = self.create_lora_model(self.base_model)
#             try:
#                 set_peft_model_state_dict(lora_model, lora_state)
#                 lora_model.base_model.head.load_state_dict(head_state)
#                 lora_models[name] = lora_model.to(self.device)
#             except Exception as e:
#                 print(f"Error loading LoRA state for domain {name}: {str(e)}")
#                 continue

#         if not lora_models:
#             raise ValueError("No valid LoRA models found. Cannot train coefficients.")


#         best_accuracy = 0
#         best_coefficients = F.softmax(coefficients, dim=0).detach().clone()

#         for epoch in range(self.config['coefficient_epochs']):
#             total_loss = 0
#             correct = 0
#             total = 0
#             for inputs, targets, _ in self.domain_loaders[self.test_domain]['coeff']:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 optimizer.zero_grad()
                
#                 outputs = sum(F.softmax(coefficients, dim=0)[i] * model(inputs) for i, model in enumerate(lora_models.values()))
                
#                 loss = self.criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
            
#             accuracy = 100. * correct / total
#             print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.domain_loaders[self.test_domain]['coeff']):.4f}, Accuracy: {accuracy:.2f}%")
#             current_coefficients = F.softmax(coefficients, dim=0)
#             print(f"Coefficients: {current_coefficients.detach().cpu().numpy()}")
            
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_coefficients = current_coefficients.detach().clone()

#         print(f"Best coefficient accuracy: {best_accuracy:.2f}%")
#         print(f"Final best coefficients: {best_coefficients.cpu().numpy()}")
#         return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))

#     def create_merged_model(self, coefficients):
#         class MergedModel(nn.Module):
#             def __init__(self, base_model, lora_adapters, coefficients, config):
#                 super().__init__()
#                 self.base_model = base_model
#                 self.lora_adapters = nn.ModuleDict()
#                 for name, (lora_state, head_state) in lora_adapters.items():
#                     lora_model = get_peft_model(base_model, LoraConfig(
#                         r=config['lora_r'],
#                         lora_alpha=config['lora_alpha'],
#                         target_modules=["qkv"],
#                         lora_dropout=config['lora_dropout'],
#                         bias="none",
#                     ))
#                     set_peft_model_state_dict(lora_model, lora_state)
#                     lora_model.base_model.head.load_state_dict(head_state)
#                     self.lora_adapters[name] = lora_model
#                 self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

#             def forward(self, x):
#                 outputs = [adapter(x) for adapter in self.lora_adapters.values()]
#                 return sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), outputs))

#         return MergedModel(self.base_model, self.lora_adapters, coefficients, self.config).to(self.device)

#     def run_baseline_experiment(self):
#         print("\nRunning Baseline experiment (train and test on test domain)")
#         baseline_model = deepcopy(self.base_model).to(self.device)
#         optimizer = optim.AdamW(baseline_model.parameters(), lr=self.config['learning_rate'])
        
#         best_val_accuracy = 0
#         best_model_state = None

#         train_loader = self.domain_loaders[self.test_domain]['coeff']
#         val_loader = self.domain_loaders[self.test_domain]['eval']

#         for epoch in range(self.config['baseline_epochs']):
#             train_loss, train_accuracy = self.train_epoch(baseline_model, train_loader, optimizer)
#             val_loss, val_accuracy = self.evaluate(baseline_model, val_loader)
            
#             print(f"Epoch {epoch+1}/{self.config['baseline_epochs']}, "
#                   f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#             print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
#             if val_accuracy > best_val_accuracy:
#                 best_val_accuracy = val_accuracy
#                 best_model_state = deepcopy(baseline_model.state_dict())

#         baseline_model.load_state_dict(best_model_state)
#         baseline_loss, baseline_accuracy = self.evaluate(baseline_model, val_loader)
#         print(f"Baseline Model - Final Test Loss: {baseline_loss:.4f}, Test Accuracy: {baseline_accuracy:.2f}%")
        
#         return baseline_accuracy

#     def summarize_single_run(self, result):
#         summary = {
#             'test_domain': result['test_domain'],
#             'domain_accuracies': {},
#             'final_accuracy': result['merged_accuracy'],
#             'coefficients': []
#         }

#         for domain, accuracy in result['domain_accuracies'].items():
#             summary['domain_accuracies'][domain] = accuracy

#         for domain, coeff in result['coefficients'].items():
#             summary['coefficients'].append(coeff)

#         print("\nSingle Run Summary:")
#         print(f"Test Domain: {summary['test_domain']}")
#         print("Domain Accuracies:")
#         for domain, accuracy in summary['domain_accuracies'].items():
#             print(f"  {domain}: {accuracy:.2f}%")
#         print(f"Final Accuracy: {summary['final_accuracy']:.2f}%")
#         print("Coefficients:", [f"{c:.4f}" for c in summary['coefficients']])

#         return summary

#     def run_experiment(self):
#         self.prepare_data()
#         print(f"\nRunning LoRA with Separate Classifiers experiment")
#         print(f"Test domain: {self.test_domain}")
#         print(f"Train domains: {self.train_domains}")

#         base_accuracy = self.run_baseline_experiment()
        
#         self.lora_adapters = {}
#         domain_accuracies = {}
#         for train_domain in self.train_domains:
#             lora_state, classifier_state, accuracy = self.train_lora_adapter(train_domain)
#             if lora_state is not None and classifier_state is not None:
#                 self.lora_adapters[train_domain] = (lora_state, classifier_state)
#                 domain_accuracies[train_domain] = accuracy
#                 print(f"{train_domain} adapter trained. Validation Accuracy: {accuracy:.2f}%")
#             else:
#                 print(f"Warning: Failed to train adapter for domain {train_domain}")

#         if not self.lora_adapters:
#             raise ValueError("No LoRA adapters were successfully trained. Cannot proceed with the experiment.")

#         self.coefficients = self.train_coefficients()

#         merged_model = self.create_merged_model(self.coefficients)
#         merged_loss, merged_accuracy = self.evaluate(merged_model, self.domain_loaders[self.test_domain]['eval'])
#         print(f"\nMerged model - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")

#         result = {
#             'test_domain': self.test_domain,
#             'train_domains': self.train_domains,
#             'base_accuracy': base_accuracy,
#             'domain_accuracies': domain_accuracies,
#             'merged_accuracy': merged_accuracy,
#             'coefficients': self.coefficients
#         }

#         # Generate and print the summary for this run
#         run_summary = self.summarize_single_run(result)

#         return result, run_summary

# def summarize_results(results):
#     summary = {
#         'base_accuracies': [],
#         'merged_accuracies': [],
#         'domain_accuracies': defaultdict(list),
#         'coefficients': defaultdict(list)
#     }

#     for result in results:
#         summary['base_accuracies'].append(result['base_accuracy'])
#         summary['merged_accuracies'].append(result['merged_accuracy'])
#         for domain, accuracy in result['domain_accuracies'].items():
#             summary['domain_accuracies'][domain].append(accuracy)
#         for domain, coeff in result['coefficients'].items():
#             summary['coefficients'][domain].append(coeff)

#     print(f"Base Model Accuracy: {np.mean(summary['base_accuracies']):.2f}% ± {np.std(summary['base_accuracies']):.2f}%")
#     print(f"Merged Model Accuracy: {np.mean(summary['merged_accuracies']):.2f}% ± {np.std(summary['merged_accuracies']):.2f}%")
    
#     print("\nDomain-specific Accuracies:")
#     for domain, accuracies in summary['domain_accuracies'].items():
#         print(f"  {domain}: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    
#     print("\nAverage Coefficients:")
#     for domain, coeffs in summary['coefficients'].items():
#         print(f"  {domain}: {np.mean(coeffs):.4f} ± {np.std(coeffs):.4f}")

#     return summary

# def main(config):
#     num_runs = config['num_runs']
#     all_results = []
#     all_summaries = []

#     experiment = Baseline4Experiment(config)

#     for run in range(num_runs):
#         print(f"\n--- Run {run + 1}/{num_runs} ---")
#         result, summary = experiment.run_experiment()
#         all_results.append(result)
#         all_summaries.append(summary)
    
#     print("\nOverall Summary of All Runs:")
#     summarize_results(all_results)

#     # If you want to print all individual run summaries at the end
#     print("\nAll Run Summaries:")
#     print({"results": all_summaries}) 




# if __name__ == "__main__":
#     config = {
#         'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/PACS',
#         'max_samples_per_class': None,
#         'base_model': 'vit_base_patch16_224',
#         'batch_size': 32,
#         'lora_r': 8,
#         'lora_alpha': 32,
#         'lora_dropout': 0.05,
#         'lora_learning_rate': 1e-4,
#         'lora_epochs': 10,
#         'coefficient_learning_rate': 1e-3,
#         'coefficient_epochs': 10,
#         'seed': 42,
#         'num_runs': 10,
#         'learning_rate': 1e-4,
#         'baseline_epochs': 10  ,
#         'num_runs': 10,
        
#     }
#     main(config)