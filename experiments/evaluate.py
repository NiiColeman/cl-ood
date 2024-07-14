# src/experiments/evaluate.py

import torch
from tqdm import tqdm
import random
from data.cl_benchmark_generator import CLBenchmarkGenerator
from models.ood_generalization_lora import OODGeneralizationLoRA

def fine_tune(model, train_loader, num_epochs, learning_rate, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(num_epochs), desc="Fine-tuning"):
        model.train()
        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize test dataset
    test_dataset = CLBenchmarkGenerator(config['datasets'][config['test_dataset']], max_samples_per_class=config['max_samples_per_class'])

    # Initialize model
    model = OODGeneralizationLoRA(config['base_model'], data=test_dataset.num_classes)
    model.load_state_dict(torch.load(config['final_model_path']))
    model = model.to(device)

    # Get all domains and randomly select domains for fine-tuning and testing
    all_domains = list(range(test_dataset.num_domains))
    num_finetune_domains = config['num_finetune_domains']
    finetune_domains = random.sample(all_domains, num_finetune_domains)
    test_domains = list(set(all_domains) - set(finetune_domains))

    print(f"\n--- Fine-tuning on {num_finetune_domains} domains of {config['test_dataset']} ---")
    for domain in finetune_domains:
        domain_data = test_dataset.get_domain_data(domain)
        train_loader = torch.utils.data.DataLoader(domain_data, batch_size=config['batch_size'], shuffle=True)
        
        fine_tune(model, train_loader, config['fine_tune_epochs'], config['fine_tune_lr'], device)

    # Evaluate on remaining test domains
    print(f"\n--- Evaluating on {len(test_domains)} held-out domains of {config['test_dataset']} ---")
    total_correct = 0
    total_samples = 0
    
    for domain in test_domains:
        test_data = test_dataset.get_domain_data(domain)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
        
        accuracy, correct, total = evaluate(model, test_loader, device)
        print(f"Accuracy on domain {domain}: {accuracy:.2f}%")
        
        total_correct += correct
        total_samples += total

    overall_accuracy = (total_correct / total_samples) * 100
    print(f"\nOverall Out-of-Distribution Accuracy on held-out domains of {config['test_dataset']}: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    import yaml
    with open('configs/experiment_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)