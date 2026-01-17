
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
from copy import deepcopy
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from data.cl_benchmark_generator import CLBenchmarkGenerator  # Import the CLBenchmarkGenerator
import timm
from torch.utils.data import random_split
from collections import defaultdict

print("50 OfficeHome")

class Baseline5Experiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.load_dataset()
        self.base_model = self.create_base_model()
        self.expert_models = {}
        self.test_domain = None
        self.train_domains = None
        self.domain_loaders = None
        self.dataset = self.load_dataset()
        self.test_domain_index=0

 

    def load_dataset(self):
        return CLBenchmarkGenerator(
        self.config['dataset_path'],
        max_samples_per_class=self.config.get('max_samples_per_class'),
        augmentation='random_crop random_flip color_jitter rotation grayscale gaussian_blur'
    )

    def create_base_model(self):
        # Implement this method based on your base model creation logic
        # For example:
        return timm.create_model(self.config['base_model'], pretrained=True, num_classes=self.dataset.num_classes)
    
    def update_head(self,model):
        for name, param in model.named_parameters():
            if 'head' in name:
                param.requires_grad = True
        
        return model


    def prepare_data(self):
        self.domains = sorted(list(set(self.dataset.domains)))  # Sort to ensure consistent ordering
        num_domains = len(self.domains)
        
        # Select the test domain based on the current index
        self.test_domain = self.domains[Baseline5Experiment.test_domain_index]
        
        # Update the index for the next run
        Baseline5Experiment.test_domain_index = (Baseline5Experiment.test_domain_index + 1) % num_domains
        
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
                train_size = int(0.10 * len(domain_data))
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
        model=self.update_head(model)
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
            

        # Merge and unload the LoRA adapter
        merged_model = model.merge_and_unload()
        merged_model = merged_model.to(self.device)
        print(f"Merged model device: {next(merged_model.parameters()).device}")
        return merged_model.cpu()  # Return the model on CPU

    def train_expert_models(self):
        for domain in self.train_domains:
            self.expert_models[domain] = self.train_and_merge_domain_model(domain)

    def train_coefficients(self):
        print("Training coefficients for weight averaging")
        coefficients = nn.Parameter(torch.ones(len(self.train_domains), device=self.device))
        optimizer = optim.Adam([coefficients], lr=self.config['coefficient_learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # Debug: Print device information
        print(f"Device: {self.device}")
        print(f"Coefficients device: {coefficients.device}")
        initial_coeffs = torch.softmax(coefficients, dim=0)
        print("Initial coefficients (after normalization):")
        for domain, coeff in zip(self.train_domains, initial_coeffs):
            print(f"  {domain}: {coeff.item():.4f}")
        # Ensure all expert models are on the correct device
        for domain in self.train_domains:
            self.expert_models[domain] = self.expert_models[domain].to(self.device)
            print(f"Expert model {domain} device: {next(self.expert_models[domain].parameters()).device}")

        for epoch in range(self.config['coefficient_epochs']):
            total_loss = 0
            for batch_idx, (inputs, labels, _) in enumerate(self.domain_loaders[self.test_domain]['coeff']):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = torch.zeros(inputs.size(0), self.base_model.num_classes, device=self.device)
                for i, domain in enumerate(self.train_domains):
                    # Debug: Print device information for each forward pass
                    if batch_idx == 0 and epoch == 0:
                        print(f"Domain {domain} - Input device: {inputs.device}, Model device: {next(self.expert_models[domain].parameters()).device}")
                    
                    self.expert_models[domain].eval()
                    with torch.no_grad():
                        domain_output = self.expert_models[domain](inputs)
                    outputs += coefficients[i] * domain_output
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.domain_loaders[self.test_domain]['coeff'])
            print(f"Coefficient Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Debug: Print final device information
        print("Final device check:")
        for domain in self.train_domains:
            print(f"Expert model {domain} device: {next(self.expert_models[domain].parameters()).device}")

        return torch.softmax(coefficients, dim=0)
    

    def naive_finetune_baseline(self):
        print("Performing naive fine-tuning on test domain")
        model = self.create_base_model().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config['num_epochs']):
            model.train()
            total_loss = 0
            for inputs, labels, _ in tqdm(self.domain_loaders[self.test_domain]['coeff'], desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.domain_loaders[self.test_domain]['coeff'])
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        return model

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
                param.copy_(avg_param)
        
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
        accuracy = accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy
    
    def run_experiment(self):
        print("Preparing data...")
        self.prepare_data()
        print(f"Test domain: {self.test_domain}")
        naive_model=self.naive_finetune_baseline()
        self.evaluate(naive_model,self.domain_loaders[self.test_domain]['eval'])
        print(f"Training domains: {self.train_domains}")

        print("\nTraining expert models...")
        domain_accuracies = {}
        for domain in self.train_domains:
            print(f"\nTraining model for domain: {domain}")
            self.expert_models[domain] = self.train_and_merge_domain_model(domain)
            
            domain_accuracy = self.evaluate(self.expert_models[domain], self.domain_loaders[self.test_domain]['eval'])
            domain_accuracies[domain] = domain_accuracy
            print(f"Accuracy of {domain} model on test domain: {domain_accuracy:.4f}")

        print("\nTraining coefficients...")
        coefficients = self.train_coefficients()
        print("Coefficient training complete.")
        print("Learned coefficients:", coefficients.tolist())

        print("\nCreating weight-averaged model...")
        final_model = self.create_weight_averaged_model(coefficients)
        final_model=self.update_head(final_model)
        print("Weight-averaged model created.")

        print("\nEvaluating final model...")
        final_accuracy = self.evaluate(final_model, self.domain_loaders[self.test_domain]['eval'])
        print(f"Final model accuracy on test domain: {final_accuracy:.4f}")

        return {
            'test_domain': self.test_domain,
            'domain_accuracies': domain_accuracies,
            'final_accuracy': final_accuracy,
            'coefficients': coefficients.tolist()
        }
    
def run_experiments_multiple_datasets(datasets_config, num_runs):
    all_results = {}
    
    for dataset_name, dataset_config in datasets_config.items():
        print(f"\n{'='*50}")
        print(f"Running experiments for dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Debug: Print the dataset_config
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
    Baseline5Experiment.test_domain_index = 0
    
    # Debug: Print the type and content of dataset_config
    print(f"Type of dataset_config: {type(dataset_config)}")
    print("Contents of dataset_config:")
    print(dataset_config)
    
    try:
        base_seed = int(dataset_config.get('seed', 42))  # Use a default seed if not provided
    except (ValueError, AttributeError) as e:
        print(f"Error getting seed from dataset_config: {str(e)}")
        base_seed = 42  # Use a default seed
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        config = dataset_config.copy() if isinstance(dataset_config, dict) else {}
        config['seed'] = base_seed + run
        
        try:
            
            experiment = Baseline5Experiment(config)
            results = experiment.run_experiment()
            dataset_results.append(results)
        except Exception as e:
            print(f"Error in run {run + 1}: {str(e)}")
    
    return dataset_results

def summarize_results(results):
    print("Debug: Type of results:", type(results))
    print("Debug: Content of results:", results)

    summary = {
        'final_accuracies': [],
        'domain_accuracies': defaultdict(list),
        'coefficients': []
    }

    if isinstance(results, dict):
        # If results is a dictionary (single run)
        results = [results]
    elif isinstance(results, str):
        print("Warning: Results is a string. Cannot summarize.")
        return None

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
    datasets_config ={ 
    'PACS': {
        'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/PACS',
        'max_samples_per_class': None,  # Adjust as needed
        'base_model': 'vit_base_patch16_224',  # Specify your base model
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
    'VLCS': {
            'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/VLCS',
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
    # 'OfficeHome': {
    #         'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/office_home',
    #         'max_samples_per_class': None,
    #         'base_model': 'vit_base_patch16_224',
    #         'batch_size': 32,
    #         'num_epochs': 10,
    #         'learning_rate': 1e-3,
    #         'lora_r': 8,
    #         'lora_alpha': 32,
    #         'lora_dropout': 0.1,
    #         'coefficient_learning_rate': 1e-1,
    #         'coefficient_epochs': 10,
    #         'seed': 42
    #     }
     'DomainNet': {
            'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/domain_net',
            'max_samples_per_class': 10,
            'base_model': 'vit_base_patch16_224',
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-5,
            'lora_r': 8,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'coefficient_learning_rate': 1,
            'coefficient_epochs': 15,
            'seed': 42
        }
    #  'SVIRO': {
    #         'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/sviro',
    #         'max_samples_per_class': None,
    #         'base_model': 'vit_base_patch16_224',
    #         'batch_size': 32,
    #         'num_epochs': 10,
    #         'learning_rate': 1e-4,
    #         'lora_r': 8,
    #         'lora_alpha': 32,
    #         'lora_dropout': 0.1,
    #         'coefficient_learning_rate': 1e-1,
    #         'coefficient_epochs': 10,
    #         'seed': 42
    #     }
    }
    
    num_runs = 10
     # Number of times to repeat the experiment for each dataset
    
    all_results = run_experiments_multiple_datasets(datasets_config, num_runs)
    overall_summary = summarize_results(all_results)
    # all_results = run_multiple_experiments(datasets_config, num_runs)

    # Save the results
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # print("\nOverall Summary:")
    # for dataset_name, results in all_results.items():
    #     print(f"\n{dataset_name}:")
    #     summarize_results(results)