import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import timm
from cl_benchmark_generator import CLBenchmarkGenerator

class DomainAdapterExperiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.load_dataset()
        self.base_model = self.create_base_model()
        self.domain_loaders = self._create_domain_loaders()

    def load_dataset(self):
        return CLBenchmarkGenerator(
            self.config['dataset_path'],
            max_samples_per_class=self.config.get('max_samples_per_class'),
            augmentation=self.config.get('augmentation')
        )

    def create_base_model(self):
        return timm.create_model(self.config['base_model'], pretrained=True, num_classes=self.dataset.num_classes)

    def _create_domain_loaders(self):
        domain_loaders = {}
        for domain in self.dataset.domains:
            domain_data = self.dataset.get_domain_data(domain)
            domain_loaders[domain] = DataLoader(domain_data, batch_size=self.config['batch_size'], shuffle=True)
        return domain_loaders

    def train_domain_adapter(self, domain):
        print(f"Training adapter for domain: {domain}")
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
            task_type=TaskType.IMAGE_CLASSIFICATION
        )
        model = get_peft_model(self.create_base_model(), lora_config)
        model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config['num_epochs']):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for inputs, labels, _ in tqdm(self.domain_loaders[domain], desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            avg_loss = total_loss / len(self.domain_loaders[domain])
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return model

    def run_experiment(self):
        os.makedirs(self.config['save_path'], exist_ok=True)
        
        for domain in self.dataset.domains:
            adapter_model = self.train_domain_adapter(domain)
            save_path = os.path.join(self.config['save_path'], f"adapter_{domain}")
            adapter_model.save_pretrained(save_path)
            print(f"Saved adapter for domain {domain} to {save_path}")

if __name__ == "__main__":
    config = {
        'dataset_path': '/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/datasets/PACS',
        'max_samples_per_class': 20,
        'augmentation': 'random_crop random_flip color_jitter',
        'base_model': 'vit_base_patch16_224',
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'save_path': './domain_adapters'
    }

    experiment = DomainAdapterExperiment(config)
    experiment.run_experiment()