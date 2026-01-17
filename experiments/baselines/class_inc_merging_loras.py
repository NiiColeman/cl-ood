from collections import defaultdict
import copy
import numpy as np
import os
import random
from tabulate import tabulate
from tqdm import tqdm
import yaml

from peft import get_peft_model, LoraConfig, PeftModel
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset

from cl_benchmark_generator import CLBenchmarkGenerator

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

def split_by_class_and_domain(dataset, c, domains):
    '''Get a dataset, a class id and a list of domains, return a subset that contains only the specified class for the spcified domains
    '''

    #idxs = np.where(dataset.targets == c and np.isin(dataset.domains, domains))
    #idxs = np.where(np.array(dataset.targets_ids) == c and np.isin(dataset.domains, domains))

    class_idxs = [i for i, t in enumerate(dataset.targets_ids) if t == c] # idxs of given class
    domain_idxs = [i for i, d in enumerate(dataset.domains) if d in domains] # idxs of given domains
    # intersection of the two - idxs of given class in the given domains
    idxs = [i for i in class_idxs if i in domain_idxs]

    return Subset(dataset, idxs)

def load_class_incremental_dataset(dataset_name, config, seed):
    dataset = CLBenchmarkGenerator(config['path'], max_samples_per_class=config.get('max_samples_per_class'))

    # create the subsets for each time step
    # each subset contains only the class for the specific time step, for the specified domains
    classes = list(range(config['num_classes']))
    domains = config['training_domains']
    ci_subsets = [split_by_class_and_domain(dataset, c, domains) for c in classes]

    return ci_subsets

def get_train_test_set(ci_subsets, proportions=[0.8, 0.2], seed=None):
    '''
    Get a list of class-incremental subsets, create train/test splits

    At the end, train_sets[i] contains the training set for the class i
    The same holds for test_sets
    '''

    train_sets, test_sets = [], []
    for subset in ci_subsets:
        # compute the lenghts of the splits
        lenghts = [int(p * len(subset)) for p in proportions]
        lenghts[-1] = len(subset) - sum(lenghts[:-1])

        # create randomly the splits
        gen = torch.Generator().manual_seed(seed)
        train_set, test_set = random_split(subset, lenghts, generator=gen)
        train_sets.append(train_set)
        test_sets.append(test_set)

    return train_sets, test_sets

def get_ci_dataloader(train_sets, test_sets, classes_per_timestep, n, batch_size):
    '''Define a generator function.
    Returns the training dataloader for current timestep, and the test loaders from timestep 0 to current timestep
    '''

    # for each timestep
    for t in range(len(classes_per_timestep)):

        # get the training set for current timestep
        # get test sets for timestep 0 to current timestep - needed to test forgetting on previous splits
        tr_set = ConcatDataset([train_sets[c] for c in classes_per_timestep[t]])
        test_sets_up_to_t = [ConcatDataset([test_sets[c] for c in classes_per_timestep[k]]) for k in range(t+1)]
        print(f'Testing on {[[train_sets[0].dataset.dataset.idx_to_class[c] for c in classes_per_timestep[k]] for k in range(t+1)]}')

        tr_loader = DataLoader(tr_set, batch_size=batch_size)
        test_loaders = [DataLoader(test_set, batch_size=batch_size) for test_set in test_sets_up_to_t]

        yield tr_loader, test_loaders

def lazy_merging(model, adapters_names):
    '''Lazy merging - merge all the trained adapters at the end.
    Merge with the 4 algorithms: linear, svd, ties, dare_ties
    Return a dictionary with the 4 merged models
    '''

    lazy_merged_models = {}
    weights = [1.0] * len(adapters_names)
    density = 0.2

    #for combination_type in ['linear', 'svd', 'ties', 'dare_ties']:
    for combination_type in ['linear']:
        merged_model = copy.deepcopy(model)
        merged_model.add_weighted_adapter(adapters_names, weights, f'merge_{combination_type}', combination_type, density=density)
        merged_model = merged_model.merge_and_unload(progressbar = True, adapter_names=[f'merge_{combination_type}'])
        lazy_merged_models[combination_type] = merged_model
        
    print('\nLazy Merging done!')

    return lazy_merged_models

### CURRENTLY NOT WORKING ###
def eager_to_adapter_merging(eager_to_adapter_models, model, dataset_name, train_domains, classes_per_timestep, timestep):
    '''Eager to adapter merging - merge the last trained adapter to an accumulator adapter
    Merge with 4 algorithms: linear, svd, ties, dare_ties
    Return a dictionary with the 4 merged models
    '''

    return eager_to_adapter_models
    
    if timestep == 0: # only one adapter, we do not merge
        print('Timestep 0 - nothing to merge')
        return eager_to_adapter_models

    # first time we merge two adapters
    # it means that 'accumulator_lora' doesn't exist, and we have to create it
    elif timestep == 1:
        print('Timestep 1 - time to create accumulator_lora')
        # at this point, we have only 2 adapters
        # that's why we fix classes_per_timestep[:2]
        adapters = [f'{dataset_name}_{train_domains}_c{c}' for c in classes_per_timestep[:2]]
    
    else: # here, we have 'accumulator_lora' and the last trained adapter
        print('Subsequent timesteps')
        adapters = [f'{dataset_name}_{train_domains}_c{classes_per_timestep[timestep]}', 'accumulator_lora']

    weights = [1.0] * len(adapters)
    density = 0.2

    # merge current two adapters in 'accumulator_lora'
    #for combination_type in ['linear', 'svd', 'ties', 'dare_ties']:
    for combination_type in ['linear']:
        merged_model = copy.deepcopy(model)
        merged_model.add_weighted_adapter(adapters, weights, f'accumulator_lora', combination_type, density=density)
        eager_to_adapter_models[combination_type] = merged_model

    # if last timestep, merge accumulator_lora to the base model for later evaluation
    if timestep == len(classes_per_timestep) - 1:
        print('\n\nLast timestep\n\n')
        for combination_type, model in eager_to_adapter_models.items():
            model = model.merge_and_unload(progressbar = True, adapter_names=f'accumulator_lora')
            eager_to_adapter_models[combination_type] = model

    print('Eager-to-Adapter Merging done!')
    return eager_to_adapter_models

def merge(merging_strategy, model, dataset_name, train_domains, adapter_names, classes_per_timestep, t):
    # merging_strategy can be 'all', 'eager_to_model', 'eager_to_adapter' or 'lazy'
    # + 'all' -> do all the merging strategies
    # + 'eager_to_model' -> merge immediately the trained adapter to the model
    # + 'eager_to_adapter' -> merge immediately the trained adapter into a single adapter
    # + 'lazy' -> get n trained adapters, merge them in the end

    merged_models = {}

    if merging_strategy == 'eager_to_model':
        # returns a dict just to be equivalent to the other cases
        print('\nEager-to-Model Merging')
        merged_models['eager_to_model'] = {'eager_to_model': model.merge_and_unload(progressbar = True, adapter_names=adapter_names)}

    elif merging_strategy == 'eager_to_adapter':
        print('\nEager-to-Adapter Merging')
        eager_to_adapter_models = {}
        eager_to_adapter_models = eager_to_adapter_merging(eager_to_adapter_models, model, dataset_name, train_domains, classes_per_timestep, t)
        merged_models['eager_to_adapter'] = eager_to_adapter_models

    elif merging_strategy == 'lazy':
        print('\nLazy Merging')
        merged_models['lazy'] = lazy_merging(model, adapter_names)
                    
    elif merging_strategy == 'all': # perform all merging strategies
        # eager_to_model
        print('\nEager-to-Model Merging')
        eager_to_model = copy.deepcopy(model) # just making sure 'model' is unchanged
        # returns a dict just to equivalent to other cases
        merged_models['eager_to_model'] = {'eager_to_model': eager_to_model.merge_and_unload(progressbar = True, adapter_names=adapter_names)}

        # eager_to_adapter
        print('\nEager-to-Adapter Merging')
        c_model = copy.deepcopy(model) # just making sure 'model' is unchanged
        eager_to_adapter_models = {}
        eager_to_adapter_models = eager_to_adapter_merging(eager_to_adapter_models, c_model, dataset_name, train_domains, classes_per_timestep, t)
        merged_models['eager_to_adapter'] = eager_to_adapter_models

        # lazy
        print('\nLazy Merging')
        c_model = copy.deepcopy(model) # just making sure 'model' is unchanged
        merged_models['lazy'] = lazy_merging(c_model, adapter_names)

    return merged_models

def test(merged_models, test_loaders, classes_per_timestep, idx_to_class, device):
    '''Evaluate all the models you get on all test loaders.
    Used to evaluate forgetting.

    Args:
        merged_models(dict): dictionary of models, resulting from different merging strategies and algorithms
        test_loaders(dict): dictionary of test DataLoaders. In a class-incremental settings, it contains all the test set up to current timestep t
    '''

    results = {}
    loss_fn = nn.CrossEntropyLoss()

    for merging_strategy, models in merged_models.items():
        results[merging_strategy] = {}

        for alg, model in models.items():
            results[merging_strategy][alg] = {}

            for i, t_loader in enumerate(test_loaders):
                loss, acc = evaluate(model, t_loader, loss_fn, device)
                print((loss, acc))
                current_classes = ''
                for c in classes_per_timestep[i]: current_classes += idx_to_class[c]+'_'
                current_classes = current_classes[:-1]

                results[merging_strategy][alg][current_classes] = (loss, acc)

    return results

def log_results(results, savedir, idx_to_class, classes_per_timestep, seed):
    '''Save to file the results.
    Using tabulate to create nice markdown logs.
    
    Args:
        results(dict): All the results. The structure is results[merging_strategy][merging_alg][classes] = (loss, accuracy)
        savedir(str): Filepath where to save the log
        seed(int): current random seed - just to create the filename
    '''

    print('\nLogging to file\n')

    def get_string_table(results, idx_to_class, classes_per_timestep):
        md_content = ''
        table_str = defaultdict(lambda: '')
        headers = ['trained_on']
        headers.append([[f'{idx_to_class[k]}' for k in c] for c in classes_per_timestep])
        breakpoint()

        for timestep in range(len(results)):
            for alg, res in results[timestep].items():
                # last classes are the training classes for current timestep
                trained_classes = [c for c in res.keys()][:-1]
                row = [[trained_classes, f'{acc:.2f}%'] for (_, acc) in res.values()]
                table_str[alg] += tabulate(row, headers=headers, tablefmt='pipe')

        for alg, t_str in table_str.items(): md_content += f'### {alg.upper()} Merging Results\n\n' + t_str + '\n'
        return md_content

    md_content = f'# Merging Algorithms in Class-Incremental Setting - Seed {seed}\n\n'
    md_content += '## Eager-to-Model (EtM) Strategy\n'
    etm_results = [r['eager_to_model'] for r in results]
    md_content += get_string_table(etm_results, idx_to_class, classes_per_timestep)

    md_content += '## Eager-to-Adapter (EtA) Strategy\n'
    eta_results = [r['eager_to_adapter'] for r in results]
    md_content += get_string_table(eta_results, idx_to_class, classes_per_timestep)

    md_content += '## Lazy-Merging (LM) Strategy'
    lazy_results = [r['lazy'] for r in results]
    md_content += get_string_table(lazy_results, idx_to_class, classes_per_timestep)

    with open(os.path.join(savedir, f'cl_merging_baselines_{seed}.md'), 'w') as fp:
        fp.write(md_content)


def main(config):
    '''
    Pick a dataset and process it in a class-incremental benchmark.
    Train a LoRA for k classes and merge them:
    + Merge at the end multiple LoRAs
    + Merge 'on the fly' when a new adapter is trained

    Measure forgetting in a 'diagonal' fashion
    '''

    for dataset_name, dataset_config in config['datasets'].items():
        print(f'Training with {dataset_name} using {dataset_config["training_domains"]} domains')

        #for seed in [1, 10, 42, 101, 3333]:
        for seed in [1]:
            # set for reproducible results
            set_seed(seed)
            device = config['device']

            # get a list of subsets, each one containing a single class
            # then, get list of train/test sets, each containing a single class
            ci_subsets = load_class_incremental_dataset(dataset_name, dataset_config, seed)
            train_sets, test_sets = get_train_test_set(ci_subsets, seed=seed)

            # define base model
            model = timm.create_model(config['base_model'], pretrained=True, num_classes=dataset_config['num_classes'])
            model.to(device)

            # stuff for training
            loss_fn = nn.CrossEntropyLoss()
            opt = AdamW(model.parameters(), lr=float(config['learning_rate']))

            # get the classes to process at each timestep
            classes = list(range(dataset_config['num_classes']))
            n = config['num_classes_per_timestep']
            batch_size = config['batch_size']
            classes_per_timestep = [classes[i:i+n] for i in range(0, len(classes), n)]

            # iterate over dataloaders per timestep
            # get training loader for current timestep
            # and test loaders from timestep 0 to current timestep - evaluate forgetting 'diagonally'
            results = [] # list of results per timestep
            all_adapter_names = []
            for t, (train_loader, test_loaders) in enumerate(get_ci_dataloader(train_sets, test_sets, classes_per_timestep, n, batch_size)):
                print(f'Timestep {t}\nTraining on class idxs {classes_per_timestep[t]}')
                print(f'Training on classes {[train_sets[0].dataset.dataset.idx_to_class[c] for c in classes_per_timestep[t]]}')

                # create lora model
                lora_config = LoraConfig(
                    r = config['lora_r'],
                    lora_alpha = config['lora_alpha'],
                    lora_dropout = config['lora_dropout'],
                    target_modules = ['qkv', 'classifier']
                )

                # if the model has already some adapters attached
                # no need to create a new PeftModel, just add a new adapter
                if isinstance(model, PeftModel):
                    model.add_adapter(f'{dataset_name}_{dataset_config["training_domains"]}_c{classes_per_timestep[t]}', lora_config)
                    model.set_adapter(f'{dataset_name}_{dataset_config["training_domains"]}_c{classes_per_timestep[t]}')
                
                else: # this is the first adapter attached
                    # create new peft model
                    model = get_peft_model(model, lora_config, adapter_name=f'{dataset_name}_{dataset_config["training_domains"]}_c{classes_per_timestep[t]}')
                
                # just saving the name of the adapter, for the subquent merging
                all_adapter_names.append(f'{dataset_name}_{dataset_config["training_domains"]}_c{classes_per_timestep[t]}')


                # explicitly tell to train the classifier
                for param in model.head.parameters(): param.requires_grad = True
                model.to(device)

                model = train(model, train_loader, loss_fn, opt, device, config['num_epochs'])
                merged_models = merge(config['merging_strategy'], model, dataset_name, dataset_config['training_domains'], all_adapter_names, classes_per_timestep, t)
                results.append(test(merged_models, test_loaders, classes_per_timestep, train_sets[0].dataset.dataset.idx_to_class, device))

            log_results(results, config['savedir'], train_sets[0].dataset.dataset.idx_to_class, classes_per_timestep, seed)

if __name__ == '__main__':
    with open('./configs/class_incremental_merging_loras.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    
    main(config)