import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import timm
from peft import get_peft_model, LoraConfig, TaskType,get_peft_model_state_dict,set_peft_model_state_dict
from tqdm import tqdm
import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
# Assuming you have this implemented
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

    def update_head(self, model):
        for name, param in model.named_parameters():
            if 'head' in name:
                param.requires_grad = True
        return model

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
        # Ensure the classifier head is trainable
        for name, param in model.named_parameters():
            if 'head' in name or 'classifier' in name or 'fc' in name:
                param.requires_grad = True
        return model

    def train_epoch(self, model, classifier, data_loader, optimizer):
        model.train()
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets, _ in tqdm(data_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            
            # Get features from the LoRA model
            features = model(inputs)
            
            # Pass features through the classifier
            outputs = classifier(features)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return total_loss / len(data_loader), 100. * correct / total
    
    def evaluate(self, model, classifier, data_loader):
        model.eval()
        classifier.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets, _ in tqdm(data_loader, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = model(inputs)
                outputs = classifier(features)
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
        
        # Create a copy of the classifier
        classifier = deepcopy(model.base_model.head).to(self.device)
        
        optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), 
                                lr=self.config['lora_learning_rate'])

        best_val_accuracy = 0
        best_lora_state = None
        best_classifier_state = None

        for epoch in range(self.config['lora_epochs']):
            train_loss, train_accuracy = self.train_epoch(model, classifier, self.domain_loaders[domain]['train'], optimizer)
            val_loss, val_accuracy = self.evaluate(model, classifier, self.domain_loaders[domain]['val'])
            
            print(f"Epoch {epoch+1}/{self.config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_lora_state = get_peft_model_state_dict(model)
                best_classifier_state = classifier.state_dict()

        return best_lora_state, best_classifier_state, best_val_accuracy

    def train_coefficients(self):
        print("\nTraining coefficients for adapter and classifier merging")
        coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device))
        coefficients.data += torch.randn_like(coefficients) * 0.05
        optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        lora_models = {}
        classifiers = {}
        for name, lora_state in self.lora_adapters.items():
            lora_model = get_peft_model(self.base_model, LoraConfig(
                r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=["qkv"],
                lora_dropout=self.config['lora_dropout'],
                bias="none",
            )).to(self.device)
            set_peft_model_state_dict(lora_model, lora_state)
            lora_models[name] = lora_model

            # Create a new classifier instance and load the stored state
            classifier = nn.Linear(self.base_model.head.in_features, self.base_model.head.out_features).to(self.device)
            classifier.load_state_dict(self.classifiers[name])
            classifiers[name] = classifier

    # ... rest of the method remains the same

        best_accuracy = 0
        best_coefficients = F.softmax(coefficients, dim=0).detach().clone()

        for epoch in range(self.config['coefficient_epochs']):
            total_loss = 0
            correct = 0
            total = 0
            for inputs, targets, _ in self.domain_loaders[self.test_domain]['coeff']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                # Get outputs from each LoRA model
                lora_outputs = [model(inputs) for model in lora_models.values()]
                
                # Apply classifiers to their respective outputs
                classifier_outputs = [classifier(output) for classifier, output in zip(classifiers.values(), lora_outputs)]
                
                # Weighted sum of classifier outputs
                outputs = sum(F.softmax(coefficients, dim=0)[i] * out for i, out in enumerate(classifier_outputs))
                
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
            
            scheduler.step()

        print(f"Best coefficient accuracy: {best_accuracy:.2f}%")
        print(f"Final best coefficients: {best_coefficients.cpu().numpy()}")
        return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))

    # def create_merged_model(self, coefficients):
    #     class MergedModel(nn.Module):
    #         def __init__(self, base_model, lora_adapters, coefficients, config):
    #             super().__init__()
    #             self.base_model = base_model
    #             self.lora_adapters = nn.ModuleDict()
    #             for name, state_dict in lora_adapters.items():
    #                 lora_model = get_peft_model(base_model, LoraConfig(
    #                     r=config['lora_r'],
    #                     lora_alpha=config['lora_alpha'],
    #                     target_modules=["qkv"],
    #                     lora_dropout=config['lora_dropout'],
    #                     bias="none",
                        
    #                 ))
    #                 lora_model.load_state_dict(state_dict, strict=False)
    #                 self.lora_adapters[name] = lora_model
    #             self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

    #         def forward(self, x):
    #             base_output = self.base_model(x)
    #             adapter_outputs = [adapter(x) for adapter in self.lora_adapters.values()]
    #             weighted_sum = sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), adapter_outputs))
    #             return base_output + weighted_sum

    #     return MergedModel(self.base_model, self.lora_adapters, coefficients, self.config).to(self.device)

    def create_merged_model(self, coefficients):
        class MergedModel(nn.Module):
            def __init__(self, base_model, lora_adapters, classifiers, coefficients, config):
                super().__init__()
                self.base_model = base_model
                self.lora_adapters = nn.ModuleDict()
                self.classifiers = nn.ModuleDict()
                for name, lora_state in lora_adapters.items():
                    lora_model = get_peft_model(base_model, LoraConfig(
                        r=config['lora_r'],
                        lora_alpha=config['lora_alpha'],
                        target_modules=["qkv"],
                        lora_dropout=config['lora_dropout'],
                        bias="none",
                    ))
                    set_peft_model_state_dict(lora_model, lora_state)
                    self.lora_adapters[name] = lora_model
                    
                    # Create a new classifier instance and load the stored state
                    classifier = nn.Linear(base_model.head.in_features, base_model.head.out_features)
                    classifier.load_state_dict(classifiers[name])
                    self.classifiers[name] = classifier
                    
                self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

            def forward(self, x):
                lora_outputs = [adapter(x) for adapter in self.lora_adapters.values()]
                classifier_outputs = [classifier(output) for classifier, output in zip(self.classifiers.values(), lora_outputs)]
                weighted_outputs = sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), classifier_outputs))
                return weighted_outputs

        return MergedModel(self.base_model, self.lora_adapters, self.classifiers, coefficients, self.config).to(self.device)


    def run_baseline_experiment(self):
        print("\nRunning Baseline experiment (train and test on test domain)")
        # Create a fresh copy of the base model for the baseline experiment
        baseline_model = deepcopy(self.base_model).to(self.device)
        optimizer = optim.AdamW(baseline_model.parameters(), lr=self.config['learning_rate'])
        
        best_val_accuracy = 0
        best_model_state = None

        # Use the 'coeff' split for training and 'eval' split for validation
        train_loader = self.domain_loaders[self.test_domain]['coeff']
        val_loader = self.domain_loaders[self.test_domain]['eval']

        for epoch in range(self.config['baseline_epochs']):
            train_loss, train_accuracy = self.train_baseline_epoch(baseline_model, train_loader, optimizer)
            val_loss, val_accuracy = self.baseline_evaluate(baseline_model, val_loader)
            
            print(f"Epoch {epoch+1}/{self.config['baseline_epochs']}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = deepcopy(baseline_model.state_dict())

        # Load the best model state and perform final evaluation
        baseline_model.load_state_dict(best_model_state,strict=False)
        baseline_loss, baseline_accuracy = self.evaluate(baseline_model, val_loader)
        print(f"Baseline Model - Final Test Loss: {baseline_loss:.4f}, Test Accuracy: {baseline_accuracy:.2f}%")
    
        return baseline_accuracy

    
    def baseline_evaluate(self, model, data_loader):
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

    def train_baseline_epoch(self, model, data_loader, optimizer):
        
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


    def run_experiment(self):
        print(f"\nRunning LoRA with Separate Classifiers experiment")
        print(f"Test domain: {self.test_domain}")
        print(f"Train domains: {self.train_domains}")

        base_accuracy = self.run_baseline_experiment()
        
        # Train LoRA adapters for each domain
        self.classifiers = {}
        for train_domain in self.train_domains:
            lora_state, classifier_state, accuracy = self.train_lora_adapter(train_domain)
            self.lora_adapters[train_domain] = lora_state
            self.classifiers[train_domain] = classifier_state  # Store the state_dict
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
        'dataset_path': '/disk2/dataset/scripts/data/PACS',
        'max_samples_per_class': 10,
        'base_model': 'vit_base_patch16_224',
        'batch_size': 32,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'lora_learning_rate': 1e-4,
        'lora_epochs': 2,
        'coefficient_learning_rate': 1e-1,
        'coefficient_epochs': 5,
        'seed': 42,
        'num_runs': 1,
        'learning_rate': 1e-4,
        'baseline_epochs': 2   
    }
    main(config)


















# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset, random_split
# import timm
# from peft import get_peft_model, LoraConfig, TaskType
# from tqdm import tqdm
# import random
# import numpy as np
# from copy import deepcopy
# from collections import defaultdict
# # Assuming you have this implemented
# from data.cl_benchmark_generator import CLBenchmarkGenerator

# class Baseline4Experiment:
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.set_seed(config['seed'])
#         self.dataset = self.load_dataset()
#         self.domains = list(set(self.dataset.domains))
#         self.test_domain = random.choice(self.domains)
#         self.train_domains = [d for d in self.domains if d != self.test_domain]
#         self.base_model = self.create_base_model()
#         self.domain_loaders = self.prepare_data_loaders()
#         self.lora_adapters = {}
#         self.criterion = nn.CrossEntropyLoss()

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

#     def update_head(self, model):
#         for name, param in model.named_parameters():
#             if 'head' in name:
#                 param.requires_grad = True
#         return model

#     def create_base_model(self):
#         return timm.create_model(
#             self.config['base_model'],
#             pretrained=True,
#             num_classes=self.dataset.num_classes
#         ).to(self.device)

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

#     def create_lora_model(self, base_model):
#         lora_config = LoraConfig(
#             r=self.config['lora_r'],
#             lora_alpha=self.config['lora_alpha'],
#             target_modules=["qkv"],
#             lora_dropout=self.config['lora_dropout'],
#             bias="none",
            
#         )
#         model = get_peft_model(base_model, lora_config)
#         # Ensure the classifier head is trainable
#         for name, param in model.named_parameters():
#             if 'head' in name or 'classifier' in name or 'fc' in name:
#                 param.requires_grad = True
#         return model

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
#         model=self.update_head(model)
#         optimizer = optim.AdamW(model.parameters(), lr=self.config['lora_learning_rate'])

#         best_val_accuracy = 0
#         best_model_state = None

#         for epoch in range(self.config['lora_epochs']):
#             train_loss, train_accuracy = self.train_epoch(model, self.domain_loaders[domain]['train'], optimizer)
#             val_loss, val_accuracy = self.evaluate(model, self.domain_loaders[domain]['val'])
            
#             print(f"Epoch {epoch+1}/{self.config['lora_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#             print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
#             if val_accuracy > best_val_accuracy:
#                 best_val_accuracy = val_accuracy
#                 best_model_state = model.state_dict()

#         model.load_state_dict(best_model_state)
#         # save model head weights

#         return best_model_state, best_val_accuracy

#     def train_coefficients(self):
#         print("\nTraining coefficients for adapter merging")
#         coefficients = nn.Parameter(torch.ones(len(self.lora_adapters), device=self.device))
#         coefficients.data += torch.randn_like(coefficients) * 0.05
#         optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

#         lora_models = {name: self.create_lora_model(self.base_model).to(self.device) for name in self.lora_adapters.keys()}
#         for name, model in lora_models.items():
#             model.load_state_dict(self.lora_adapters[name], strict=False)

#         best_accuracy = 0
#         best_coefficients = F.softmax(coefficients, dim=0).detach().clone()  # Initialize with current coefficients

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
            
#             scheduler.step()

#         print(f"Best coefficient accuracy: {best_accuracy:.2f}%")
#         print(f"Final best coefficients: {best_coefficients.cpu().numpy()}")
#         return dict(zip(self.lora_adapters.keys(), best_coefficients.cpu().numpy()))

#     def create_merged_model(self, coefficients):
#         class MergedModel(nn.Module):
#             def __init__(self, base_model, lora_adapters, coefficients, config):
#                 super().__init__()
#                 self.base_model = base_model
#                 self.lora_adapters = nn.ModuleDict()
#                 for name, state_dict in lora_adapters.items():
#                     lora_model = get_peft_model(base_model, LoraConfig(
#                         r=config['lora_r'],
#                         lora_alpha=config['lora_alpha'],
#                         target_modules=["qkv"],
#                         lora_dropout=config['lora_dropout'],
#                         bias="none",
                        
#                     ))
#                     lora_model.load_state_dict(state_dict, strict=False)
#                     self.lora_adapters[name] = lora_model
#                 self.coefficients = nn.Parameter(torch.tensor(list(coefficients.values())))

#             def forward(self, x):
#                 base_output = self.base_model(x)
#                 adapter_outputs = [adapter(x) for adapter in self.lora_adapters.values()]
#                 weighted_sum = sum(coeff * out for coeff, out in zip(F.softmax(self.coefficients, dim=0), adapter_outputs))
#                 return base_output + weighted_sum

#         return MergedModel(self.base_model, self.lora_adapters, coefficients, self.config).to(self.device)

#     def run_baseline_experiment(self):
#         print("\nRunning Baseline experiment (train and test on test domain)")
#         # Create a fresh copy of the base model for the baseline experiment
#         baseline_model = deepcopy(self.base_model).to(self.device)
#         optimizer = optim.AdamW(baseline_model.parameters(), lr=self.config['learning_rate'])
        
#         best_val_accuracy = 0
#         best_model_state = None

#         # Use the 'coeff' split for training and 'eval' split for validation
#         train_loader = self.domain_loaders[self.test_domain]['coeff']
#         val_loader = self.domain_loaders[self.test_domain]['eval']

#         for epoch in range(self.config['baseline_epochs']):
#             train_loss, train_accuracy = self.train_epoch(baseline_model, train_loader, optimizer)
#             val_loss, val_accuracy = self.evaluate(baseline_model, val_loader)
            
#             print(f"Epoch {epoch+1}/{self.config['baseline_epochs']}, "
#                 f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#             print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
#             if val_accuracy > best_val_accuracy:
#                 best_val_accuracy = val_accuracy
#                 best_model_state = deepcopy(baseline_model.state_dict())

#         # Load the best model state and perform final evaluation
#         baseline_model.load_state_dict(best_model_state)
#         baseline_loss, baseline_accuracy = self.evaluate(baseline_model, val_loader)
#         print(f"Baseline Model - Final Test Loss: {baseline_loss:.4f}, Test Accuracy: {baseline_accuracy:.2f}%")
    
#         return baseline_accuracy

#     def run_experiment(self):
#         print(f"\nRunning LoRA with Separate Classifiers experiment")
#         print(f"Test domain: {self.test_domain}")
#         print(f"Train domains: {self.train_domains}")

#         base_accuracy = self.run_baseline_experiment()
#         # Train LoRA adapters for each domain
#         for train_domain in self.train_domains:
#             lora_state, accuracy = self.train_lora_adapter(train_domain)
#             self.lora_adapters[train_domain] = lora_state
#             print(f"{train_domain} adapter trained. Validation Accuracy: {accuracy:.2f}%")

#         # Train coefficients
#         self.coefficients = self.train_coefficients()

#         # Create and evaluate merged model
#         merged_model = self.create_merged_model(self.coefficients)
#         merged_loss, merged_accuracy = self.evaluate(merged_model, self.domain_loaders[self.test_domain]['eval'])
#         print(f"\nMerged model - Test Loss: {merged_loss:.4f}, Test Accuracy: {merged_accuracy:.2f}%")

#         # Evaluate base model for comparison
#         # base_loss, base_accuracy = self.evaluate(self.base_model, self.domain_loaders[self.test_domain]['eval'])
#         # print(f"Base model - Test Loss: {base_loss:.4f}, Test Accuracy: {base_accuracy:.2f}%")

#         return {
#             'test_domain': self.test_domain,
#             'train_domains': self.train_domains,
#             'base_accuracy': base_accuracy,
#             'merged_accuracy': merged_accuracy,
#             'coefficients': self.coefficients
#         }

# def main(config):
#     num_runs = config.get('num_runs', 1)
#     all_results = []

#     for run in range(num_runs):
#         print(f"\n--- Run {run + 1}/{num_runs} ---")
#         experiment = Baseline4Experiment(config)
#         result = experiment.run_experiment()
#         all_results.append(result)
    
#     print("\nSummary of All Runs:")
#     base_accuracies = [run['base_accuracy'] for run in all_results]
#     merged_accuracies = [run['merged_accuracy'] for run in all_results]
    
#     print(f"Base Model Accuracy: {np.mean(base_accuracies):.2f}% ± {np.std(base_accuracies):.2f}%")
#     print(f"Merged Model Accuracy: {np.mean(merged_accuracies):.2f}% ± {np.std(merged_accuracies):.2f}%")
    
#     print("Average Coefficients:")
#     avg_coefficients = defaultdict(list)
#     for run in all_results:
#         for domain, coeff in run['coefficients'].items():
#             avg_coefficients[domain].append(coeff)
#     for domain, coeffs in avg_coefficients.items():
#         print(f"  {domain}: {np.mean(coeffs):.4f} ± {np.std(coeffs):.4f}")

# if __name__ == "__main__":
#     config = {
#         'dataset_path': '/disk2/dataset/scripts/data/PACS',
#         'max_samples_per_class': 100,
#         'base_model': 'vit_base_patch16_224',
#         'batch_size': 32,
#         'lora_r': 8,
#         'lora_alpha': 32,
#         'lora_dropout': 0.1,
#         'lora_learning_rate': 1e-4,
#         'lora_epochs': 5,
#         'coefficient_learning_rate': 1e-1,
#         'coefficient_epochs': 5,
#         'seed': 42,
#         'num_runs': 1,
#         'learning_rate': 1e-4,
#         'baseline_epochs': 5   
#     }
#     main(config)