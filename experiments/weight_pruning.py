import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import timm
from data.cl_benchmark_generator import CLBenchmarkGenerator
from torch.utils.data import random_split
import random


class ImprovedBaseline5Experiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.load_dataset()
        self.base_model = self.create_base_model()
        self.expert_models = {}
        self.test_domain = None
        self.train_domains = None
        self.domain_loaders = None

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
        self.test_domain = random.choice(self.domains)
        self.train_domains = [d for d in self.domains if d != self.test_domain]
        self.domain_loaders = self._create_domain_loaders()
        print(f"Selected test domain: {self.test_domain}")
        print(f"Training domains: {self.train_domains}")

    def _create_domain_loaders(self):
        domain_loaders = {}
        for domain in self.domains:
            domain_data = self.dataset.get_domain_data(domain)
            if domain == self.test_domain:
                train_size = int(0.5 * len(domain_data))
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

    def prune_lora_weights(self, model, pruning_threshold):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    mask = torch.abs(param.data) > pruning_threshold
                    param.data *= mask
        return model

    def train_and_merge_domain_model(self, domain):
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

        # Prune and merge the LoRA adapter
        pruning_threshold = self.config['pruning_threshold']
        pruned_model = self.prune_lora_weights(model, pruning_threshold)
        merged_model = pruned_model.merge_and_unload()
        return merged_model.cpu()

    def train_expert_models(self):
        for domain in self.train_domains:
            self.expert_models[domain] = self.train_and_merge_domain_model(domain)

    def train_coefficients(self):
        print("Training coefficients for weight averaging")
        coefficients = nn.Parameter(torch.ones(len(self.train_domains), device=self.device))
        optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()

        # Move all expert models to the device
        for domain in self.train_domains:
            self.expert_models[domain] = self.expert_models[domain].to(self.device)

        for epoch in range(self.config['coefficient_epochs']):
            total_loss = 0
            for inputs, labels, _ in self.domain_loaders[self.test_domain]['coeff']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = torch.zeros(inputs.size(0), self.base_model.num_classes, device=self.device)
                for i, domain in enumerate(self.train_domains):
                    self.expert_models[domain].eval()
                    with torch.no_grad():
                        domain_output = self.expert_models[domain](inputs)
                    outputs += F.softmax(coefficients[i]) * domain_output
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.domain_loaders[self.test_domain]['coeff'])
            print(f"Coefficient Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)

        # Move expert models back to CPU to free up GPU memory
        for domain in self.train_domains:
            self.expert_models[domain] = self.expert_models[domain].cpu()

        return F.softmax(coefficients, dim=0)

    def create_weight_averaged_model(self, coefficients):
        print("Creating weight-averaged model")
        avg_model = self.create_base_model()
        avg_model.to(self.device)
        
        with torch.no_grad():
            for name, param in avg_model.named_parameters():
                avg_param = torch.zeros_like(param, device=self.device)
                for i, domain in enumerate(self.train_domains):
                    self.expert_models[domain].to(self.device)
                    expert_param = self.expert_models[domain].state_dict()[name].to(self.device)
                    avg_param += coefficients[i] * expert_param
                    self.expert_models[domain].cpu()  # Move back to CPU after use
                param.copy_(avg_param)
        
        return avg_model

    class EnsembleMergedModels(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = nn.ModuleList(models)
        
        def forward(self, x):
            outputs = [model(x) for model in self.models]
            return torch.mean(torch.stack(outputs), dim=0)

    def create_ensemble_model(self, pruning_thresholds):
        merged_models = []
        for threshold in pruning_thresholds:
            self.config['pruning_threshold'] = threshold
            self.train_expert_models()
            coefficients = self.train_coefficients()
            merged_model = self.create_weight_averaged_model(coefficients)
            merged_models.append(merged_model)
        
        return self.EnsembleMergedModels(merged_models)

    def compute_weight_importance(self, model, data_loader):
        model.eval()
        importance = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        criterion = nn.CrossEntropyLoss()

        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    importance[name] += torch.abs(param.grad)

        return importance

    def create_importance_weighted_model(self):
        base_model = self.create_base_model().to(self.device)
        importance_sum = {name: torch.zeros_like(param) for name, param in base_model.named_parameters()}

        for domain in self.train_domains:
            importance = self.compute_weight_importance(self.expert_models[domain], self.domain_loaders[self.test_domain]['coeff'])
            for name in importance_sum:
                importance_sum[name] += importance[name]

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                weighted_param = torch.zeros_like(param)
                for domain in self.train_domains:
                    expert_param = self.expert_models[domain].state_dict()[name]
                    weight = importance_sum[name] / (len(self.train_domains) * importance_sum[name].sum())
                    weighted_param += weight * expert_param
                param.copy_(weighted_param)

        return base_model

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

        print("\nTraining expert models...")
        self.train_expert_models()

        print("\nCreating final models...")
        pruned_model = self.create_weight_averaged_model(self.train_coefficients())
        ensemble_model = self.create_ensemble_model([1e-3, 1e-4, 1e-5])
        importance_weighted_model = self.create_importance_weighted_model()

        print("\nEvaluating final models...")
        pruned_accuracy = self.evaluate(pruned_model, self.domain_loaders[self.test_domain]['eval'])
        ensemble_accuracy = self.evaluate(ensemble_model, self.domain_loaders[self.test_domain]['eval'])
        importance_weighted_accuracy = self.evaluate(importance_weighted_model, self.domain_loaders[self.test_domain]['eval'])

        return {
            'test_domain': self.test_domain,
            'pruned_accuracy': pruned_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'importance_weighted_accuracy': importance_weighted_accuracy
        }

# Usage
if __name__ == "__main__":
    config = {
        'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/PACS',
        'base_model': 'vit_base_patch16_224',
        'max_samples_per_class': 20,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'coefficient_learning_rate': 0.01,
        'coefficient_epochs': 10,
        'pruning_threshold': 1e-3,
        'seed': 42
    }

    experiment = ImprovedBaseline5Experiment(config)
    results = experiment.run_experiment()
    print(results)