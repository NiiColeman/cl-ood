# src/experiments/baseline_experiments.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from cl_benchmark_generator import CLBenchmarkGenerator
import timm
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, AutoPeftModel, load_peft_weights, set_peft_model_state_dict, get_peft_model_state_dict
from tqdm import tqdm
import logging
import os
import random
import yaml
import csv
import numpy as np

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return model

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def set_seed(seed):
    print(f'Setting seed to {seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_lora_model(dataset_name, test_domain, dataset_config, global_config):
    '''Train (or load) a lora adapter on each training domain.
    Returns a PeftModel with a trained lora adapter for each training domain
    '''

    # load the dataset
    dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
    print(f'Dataset len: {len(dataset)}')

    # Get all the unique domains
    domains = list(set(dataset.domains))
    train_domains = [d for d in domains if d != test_domain]

    print(f'Number of domains: {len(domains)}')
    print(f'Test domain: {test_domain}')
    print(f'Training domains: {train_domains}')

    # Create the model
    model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    device = global_config['device']

    # Train (or load) the adapters
    for train_domain in train_domains:
        # load
        ######################
        #### Do not works ####
        ######################

        # train
        # Define the training loader
        train_idxs = [i for i,d in enumerate(dataset.domains) if d == train_domain]
        train_subset = Subset(dataset, train_idxs)
        train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)

        # Create the LoRA model
        lora_config = LoraConfig(
                r = global_config['lora_r'],
                lora_alpha = global_config['lora_alpha'],
                target_modules = ['qkv', 'classifier'],
                lora_dropout = global_config['lora_dropout'])

        # if the variable 'lora_model' do not exists
        # it is the first time an adapter is attached to the model, hence we need to create the PeftModel
        if not 'lora_model' in locals():
            lora_model = get_peft_model(model, lora_config, adapter_name=f'{dataset_name}_{train_domain}_lora')

        else: # the PeftModel already exists, just add a new adapter
            lora_model.add_adapter(f'{dataset_name}_{train_domain}_lora', lora_config)
            lora_model.set_adapter(f'{dataset_name}_{train_domain}_lora')

        # explicity tell to train the classifier
        for param in lora_model.head.parameters(): param.requires_grad=True

        lora_model.to(device)

        # Train the LoRA model
        opt = optim.AdamW(lora_model.parameters(), lr = float(global_config['learning_rate']))
        lora_model = train(lora_model, train_loader, loss_fn, opt, device, global_config['num_epochs'])

    return lora_model

def get_merging_weights(method, num_adapters, base_model=None, dataset=None, n_shots=None, test_domain=None, device='cuda'):
    '''
    Returns the weights to use when merging loras.
    'fixed' method returns an hardcoded list of weights.
    'distance' method computes the distance between test_domain average representation and each train_domain average representation.
    Such distance is returned as the merging weights. Optionally, such weights can be normalized to sum to 1.

    Example of 'distance' method:
    PACS dataset, cartoon as test domain -> returns [sim(repr_cartoon, repr_art), sim(repr_cartoon, repr_photo), sim(repr_cartoon, repr_sketch)]
    where sim() is the measure of similarity, e.g. cosine similarity

    Parameters:
    ----------
    method: str
        The method to compute the weights. Should be in ('fixed', 'distance')
    num_adapters: int
        The number of adapters, equal to the number of weights to be returned
    base_model:
        The pretrained model, without any adapter. Used when method=='distance' to compute the representations of train and test domains
    n_shots: int
        The number of samples to take from each domain. Needed to compute the representation of train and test domains
    dataset:
        The dataset from which to extract the samples. Needed to compute the representation of train and test domains
    test_domain: str
        The name of the test domain. Needed to know the distances to compute
    '''

    if method == 'fixed': return [1.0 for _ in range(num_adapters)]

    # 1. Extract n_shots samples from each domain
    # 2. Forward pass on the samples for each domain. Retrieve the list of representations before classifier and take the average for each domain
    # 3. Compute the distances between avg representations -  use cosine similarity
    elif method == 'distance':

        # 1. Extract n_shots samples from each domain at random
        # Save the dataloaders for later
        loaders = {}
        for domain in dataset.domains:
            total_idxs = [i for i, d in enumerate(dataset.domains) if d == domain]
            samples_idxs = random.choices(total_idxs, k=n_shots)
            subset = Subset(dataset, samples_idxs)

            loaders[domain] = DataLoader(subset, batch_size=len(subset))

        # 2. Forward pass to compute representations
        base_model.reset_classifier(0) # remove classifier to get pooled last layer features
        base_model.to(device)
        base_model.eval()

        # forward pass
        with torch.no_grad():
            representations = {}
            for domain, loader in loaders.items():

                outs = []
                for inputs, targets, _ in loader:
                    inputs = inputs.to(device)
                    outs.append(base_model(inputs))
                #print(f'Shape of outputs: [{len(outs)}, {outs[0].shape}]')

                representations[domain] = torch.stack(outs).squeeze().mean(dim=0) # compute the average of the representations for the given domain samples
                #print(f'Shape of outputs average: {representations[domain].shape}')

        
        # 3. Compute cosine similarity
        weights = {}
        for domain in sorted(dataset.domains):
            if domain != test_domain:
                weights[domain] = nn.functional.cosine_similarity(representations[test_domain], representations[domain], dim=0)

        print(f'Weights before normalization: {weights}')

        # Softmax to normalize the weights
        norm_weights = nn.functional.softmax(torch.Tensor(list(weights.values())))
        weights = {d: norm_weights[i] for i, d in enumerate(weights.keys())}

        return weights

def merging_loras_baseline(dataset_name, dataset_config, global_config, seed):
    print(f"Running baseline experiments for dataset: {dataset_name}")

    results = []
    log_weights = []

    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the dataset
    dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
    print(f'Dataset len: {len(dataset)}')
    
    # Get all unique domains
    domains = list(set(dataset.domains))
    num_domains = len(domains)
    print(f"Total number of domains: {num_domains}")

    # select each domain as test_domain iteratively
    for test_domain in domains:
        train_domains = [d for d in domains if d != test_domain]
        print(f"Test domain: {test_domain}")
        print(f"Train domains: {train_domains}")

        # Create model
        model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
        base_model = copy.deepcopy(model) # needed to compute merging weights
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=float(global_config['learning_rate']))

        # Merging_LoRAs Baseline
        # Train (or load from memory) an adapter for each train domain
        # Merge with different algorithms the adapters, to get a single adapter
        # Merge the resulting adapter to the base model -> zero-shot test on the target domain

        # 1. Train (or load) an adapter for each train domain
        for train_domain in train_domains:

            ###################
            ######LOADING######
            ##NOT WORKING NOW##
            ###################
            '''
            # load the adapter from memory, if path is provided
            ada_path = global_config.get('saved_adapters', {}).get(dataset_name, {}).get(train_domain, None)
            if ada_path:
                # if the model has already some adapter
                if 'lora_model' in locals() and isinstance(lora_model, PeftModel): lora_model.load_adapter(ada_path, f'{dataset_name}_{train_domain}_lora')
                else: # it's the first time the model gets an adapter
                    lora_model = PeftModel.from_pretrained(model, ada_path, adapter_name=f'{dataset_name}_{train_domain}_lora')
    
                print(f'Loading the adapter for {train_domain} domain from {ada_path}')
            else: # train the adapter
            '''

            # train the adapters on train_domain
            print(f'Training on domain: {train_domain}')
            train_indices = [i for i, d in enumerate(dataset.domains) if d == train_domain]
            train_subset = Subset(dataset, train_indices)
            train_loader = DataLoader(train_subset, batch_size=global_config['batch_size'], shuffle=True)
    
            # Create LoRA adapter
            lora_config = LoraConfig(
                r = global_config['lora_r'],
                lora_alpha = global_config['lora_alpha'],
                target_modules = ["qkv", "classifier"],
                lora_dropout = global_config['lora_dropout'],)
            
            # the variable 'lora_model' do not exists -> first time an adapter gets attached to the model
            # hence, we need to create the PeftModel
            if not 'lora_model' in locals():
                lora_model = get_peft_model(model, lora_config, adapter_name=f'{dataset_name}_{train_domain}_lora')

            # else, if the model has already some adapter, simply add a new one
            else:
                assert isinstance(lora_model, PeftModel), 'Your lora_model is not a PeftModel'
                lora_model.add_adapter(f'{dataset_name}_{train_domain}_lora', lora_config)
                lora_model.set_adapter(f'{dataset_name}_{train_domain}_lora')

            
            for param in lora_model.head.parameters(): param.requires_grad=True
            lora_model = lora_model.to(device)

            # Train with LoRA
            optimizer = optim.AdamW(lora_model.parameters(), lr=float(global_config['learning_rate']))
            lora_model = train(lora_model, train_loader, criterion, optimizer, device, global_config['num_epochs'])
            #lora_model.save_pretrained(f'/leonardo_scratch/fast/IscrC_FoundCL/projects/cl-collab/ModelRatatouille/lquarant/cl-ood/saved_models')
            #torch.save(lora_model.state_dict())

        # 2. Merge LoRAs with different algorithms
        adapters = [f'{dataset_name}_{train_domain}_lora' for train_domain in sorted(train_domains)]
        #weights = [0.5 for _ in range(len(adapters))]
        weights = get_merging_weights(global_config.get('merging_weights_method', 'fixed'), len(adapters), base_model, dataset, global_config.get('n_shots'), test_domain, device)
        density = 0.2

        print(adapters)
        print(weights)

        for combination_type in ['linear', 'svd', 'ties', 'dare_ties']:
        #for combination_type in ['linear', 'ties']:
            if isinstance(weights, dict):
                weights = [w.item() for w in weights.values()]

            merged_model = copy.deepcopy(lora_model)
            merged_model.add_weighted_adapter(adapters, weights, f'merge_{combination_type}', combination_type, density=density) 
            merged_model = merged_model.merge_and_unload(progressbar=True, adapter_names=[f'merge_{combination_type}']) # merge the final adapter with the model

            # 3. Evaluate on test domain
            print(f'Using {combination_type} combination type')
            print(f"Evaluating final model on test {test_domain} domain")

            test_indices = [i for i, d in enumerate(dataset.domains) if d == test_domain]
            test_subset = Subset(dataset, test_indices)
            test_loader = DataLoader(test_subset, batch_size=global_config['batch_size'], shuffle=True)

            test_loss, test_accuracy = evaluate(merged_model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            print('---')

            results.append({'seed': seed, 'test_domain': test_domain, 'combination_type': combination_type, 'test_loss': f'{test_loss:.4f}', 'test_accuracy': test_accuracy})

        log_weights.append({'seed': seed, 'test_domain': test_domain, 'merging_weights': weights})

    return results, log_weights

# debugging function
# load a trained adapter and test it on its training set
def _test_adapter(dataset_name, dataset_config, global_config):
    print('\n\n\n################\nDEBUGGING\n##########\n\n\n')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Load the dataset
    dataset = CLBenchmarkGenerator(dataset_config['path'], max_samples_per_class=global_config.get('max_samples_per_class'))
    domains = list(set(dataset.domains))

    for test_domain in domains:
        print(f'Test domain: {test_domain}')
        idxs = [i for i, d in enumerate(dataset.domains) if d == test_domain]
        subset = Subset(dataset, idxs)
        loader = DataLoader(subset, batch_size=global_config['batch_size'], shuffle=True)
  
        # Create model
        model = timm.create_model(global_config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
        model = model.to(device)

        # Load the adapter
        print(dataset_name)
        print(test_domain)
        ada_path = global_config.get('saved_adapters', {}).get(dataset_name, {}).get(test_domain, None)
        print(f'Path to the adapter: {ada_path}')
        #lora_model = PeftModel.from_pretrained(model, ada_path) #adapter_name=f'{dataset_name}_{test_domain}_lora')
        #lora_model.load_adapter(ada_path, adapter_name=f'{dataset_name}_{test_domain}_lora')
        #lora_model.set_adapter(f'{dataset_name}_{test_domain}_lora')

        lora_weights = load_peft_weights(ada_path)
        set_peft_model_state_dict(model, lora_weights)

        model.to(device)
        print(f'Loaded model from {ada_path}')

        test_loss, test_accuracy = evaluate(model, loader, nn.CrossEntropyLoss(), device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print('\n\n')

def test_loading_ada(config):
    for dataset_name, dataset_config in config['datasets'].items():
        _test_adapter(dataset_name, dataset_config, config)


def main(config):
    for dataset_name, dataset_config in config['datasets'].items():
        results = []
        weights = []

        for seed in [1, 10, 42, 101, 3333]:
            result, weight = merging_loras_baseline(dataset_name, dataset_config, config, seed)
            #result = _test_adapter(dataset_name, dataset_config, config)
            results.append(result)
            weights.append(weight)

        fields = ['seed', 'test_domain', 'combination_type', 'test_loss', 'test_accuracy']
        with open(f'results_{dataset_name}_cl.csv', 'w+') as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
    
            for r in results:
                writer.writerows(r)

        # compute avg and std for each test domain
        domains_set = set([r['test_domain'] for line in results for r in line])
        domains = list(domains_set)
        summary_res = []
        for domain in domains:
            accs = [r['test_accuracy'] for line in results for r in line if r['test_domain'] == domain]
            avg = np.mean(accs)
            std = np.std(accs)
            
            summary_res.append({'test_domain': domain, 'avg': avg, 'std': std})

        fields = ['test_domain', 'avg', 'std']
        with open(f'summary_results_{dataset_name}.csv', 'w+') as fp:
            writer = csv.DictWriter(fp, fieldnames=summary_res[0].keys())
            writer.writeheader()
            writer.writerows(summary_res)

        # save the merging weights
        fields = ['seed', 'test_domain', 'merging_weights']
        with open(f'weights_{dataset_name}.csv', 'w+') as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()

            for w in weights:
                writer.writerows(w)

    print('Done!')
    
if __name__ == "__main__":
    with open('/leonardo_scratch/fast/IscrC_FoundCL/projects/cl-collab/ModelRatatouille/lquarant/cl-ood/configs/merging_loras.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
    #test_loading_ada(config)



