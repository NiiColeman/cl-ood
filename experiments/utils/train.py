# src/experiments/train.py

from data.cl_benchmark_generator import CLBenchmarkGenerator
from models.ood_generalization_lora import OODGeneralizationLoRA
import torch
from tqdm import tqdm
import random

def main(config):
    # Initialize datasets
    datasets = {
        name: CLBenchmarkGenerator(path, max_samples_per_class=config['max_samples_per_class'])
        for name, path in config['datasets'].items()
    }

    # Get class counts for each dataset
    dataset_class_counts = {name: dataset.num_classes for name, dataset in datasets.items()}

    # Initialize model
    model = OODGeneralizationLoRA(
        base_model_name=config['base_model'],
        dataset_class_counts=dataset_class_counts,
        lora_config=config['lora_config']
    )

    # Train on auxiliary datasets
    total_datasets = len(config['auxiliary_datasets'])
    for dataset_idx, dataset_name in enumerate(config['auxiliary_datasets'], 1):
        print(f"\n--- Training on dataset {dataset_name} ({dataset_idx}/{total_datasets}) ---")
        print(f"Number of classes: {dataset_class_counts[dataset_name]}")
        dataset = datasets[dataset_name]
        model.train_auxiliary(
            dataset,
            dataset_name,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            per_domain_adapter=config['per_domain_adapter']
        )

    # Save model after auxiliary training
    torch.save(model.base_model.state_dict(), config['auxiliary_model_path'])
    print("\nAuxiliary training completed. Model saved.")

    # Fine-tune on test dataset
    test_dataset_name = config['test_dataset']
    test_dataset = datasets[test_dataset_name]
    
    # Select domains for fine-tuning and testing
    all_domains = list(range(test_dataset.num_domains))
    num_finetune_domains = config['num_finetune_domains']
    finetune_domains = random.sample(all_domains, num_finetune_domains)
    test_domain = random.choice(list(set(all_domains) - set(finetune_domains)))

    print(f"\n--- Fine-tuning on {num_finetune_domains} domains of {test_dataset_name} ---")
    for domain in finetune_domains:
        domain_data = test_dataset.get_domain_data(domain)
        train_loader = torch.utils.data.DataLoader(domain_data, batch_size=config['batch_size'], shuffle=True)
        
        model.fine_tune(
            train_loader,
            num_epochs=config['fine_tune_epochs'],
            learning_rate=config['fine_tune_lr'],
            num_classes=dataset_class_counts[test_dataset_name]
        )

    # Save fine-tuned model
    torch.save(model.base_model.state_dict(), config['final_model_path'])
    print("Fine-tuning completed. Final model saved.")

    # Evaluate on held-out domain
    print(f"\n--- Evaluating on held-out domain {test_domain} of {test_dataset_name} ---")
    test_data = test_dataset.get_domain_data(test_domain)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    
    accuracy = model.evaluate(test_loader, dataset_class_counts[test_dataset_name])
    print(f"Out-of-Distribution Accuracy on domain {test_domain}: {accuracy:.2f}%")

if __name__ == "__main__":
    import yaml
    with open('configs/experiment_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)