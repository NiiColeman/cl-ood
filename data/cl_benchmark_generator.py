# src/data/cl_benchmark_generator.py

import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict
# import logging
# logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

class CLBenchmarkGenerator(Dataset):
    def __init__(self, dataset_path: str, max_samples_per_class: int = None):
        self.dataset_path = dataset_path
        self.max_samples_per_class = max_samples_per_class
        self.image_paths, self.targets, self.domains = self._load_data()
        self.num_samples = len(self.image_paths)
        self.num_domains = len(set(self.domains))
        
        unique_classes = sorted(set(self.targets))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.num_classes = len(self.class_to_idx)
        self.domain_to_indices = self._create_domain_indices()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_data(self):
        image_paths, targets, domains = [], [], []
        corrupted_images = []
        for domain in os.listdir(self.dataset_path):
            domain_path = os.path.join(self.dataset_path, domain)
            if not os.path.isdir(domain_path):
                continue
            for class_name in os.listdir(domain_path):
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                class_images = os.listdir(class_path)
                if self.max_samples_per_class:
                    class_images = class_images[:self.max_samples_per_class]
                for image_name in class_images:
                    image_path = os.path.join(class_path, image_name)
                    try:
                        with Image.open(image_path) as img:
                            img.verify()  # Verify that it's a valid image
                        image_paths.append(image_path)
                        targets.append(class_name)
                        domains.append(domain)
                    except (IOError, SyntaxError) as e:
                        corrupted_images.append(image_path)
                        print(f"Corrupted image found and skipped: {image_path}")
        
        if corrupted_images:
            print(f"Total corrupted images found and skipped: {len(corrupted_images)}")
        
        return image_paths, targets, domains

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except (IOError, OSError) as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image and the correct label if we can't load the image
            image = torch.zeros((3, 224, 224))
        
        target = self.class_to_idx[self.targets[idx]]
        domain = self.domains[idx]
        return image, torch.tensor(target, dtype=torch.long), domain

    def __len__(self):
        return self.num_samples

    def update_class_indices(self, global_class_to_idx):
        self.class_to_idx = global_class_to_idx
        self.num_classes = len(self.class_to_idx)

    def create_incremental_split(self, n_domains: int) -> List[Tuple[Subset, Subset]]:
        tasks = []
        for domain_id in range(n_domains):
            train_indices = [i for i, d in enumerate(self.domains) if self.domain_to_id[d] == domain_id]
            test_indices = [i for i, d in enumerate(self.domains) if self.domain_to_id[d] != domain_id]
            train_subset = Subset(self, train_indices)
            test_subset = Subset(self, test_indices)
            tasks.append((train_subset, test_subset))
        return tasks
    
    def get_domain_data(self, domain_id):
        domain_indices = [i for i, d in enumerate(self.domains) if d == domain_id]
        return Subset(self, domain_indices)

    def create_data_loaders(self, tasks: List[Tuple[Subset, Subset]], batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader]]:
        return [
            (DataLoader(train_subset, batch_size=batch_size, shuffle=True),
             DataLoader(test_subset, batch_size=batch_size, shuffle=False))
            for train_subset, test_subset in tasks
        ]


    def print_samples_per_class_per_domain(self):
        domain_class_counts = defaultdict(lambda: defaultdict(int))
        for domain, target in zip(self.domains, self.targets):
            domain_class_counts[domain][target] += 1
        
        for domain, class_counts in domain_class_counts.items():
            print(f"\nDomain: {domain}")
            for class_name, count in class_counts.items():
                print(f"  Class {class_name}: {count} samples")
            print(f"  Total samples: {sum(class_counts.values())}")

    def _create_domain_indices(self):
        domain_to_indices = {}
        for idx, domain in enumerate(self.domains):
            if domain not in domain_to_indices:
                domain_to_indices[domain] = []
            domain_to_indices[domain].append(idx)
        return domain_to_indices

    def get_domain_data(self, domain):
        if isinstance(domain, int):
            domain = list(self.domain_to_indices.keys())[domain]
        
        if domain not in self.domain_to_indices:
            raise ValueError(f"Domain {domain} not found in the dataset.")
        
        indices = self.domain_to_indices[domain]
        return Subset(self, indices)



























# src/data/cl_benchmark_generator.py

# import os
# import torch
# from torch.utils.data import Dataset, Subset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from typing import List, Tuple, Dict
# from collections import defaultdict

# class CLBenchmarkGenerator(Dataset):
#     def __init__(self, dataset_path: str, max_samples_per_class: int = None):
#         self.dataset_path = dataset_path
#         self.max_samples_per_class = max_samples_per_class
#         self.image_paths, self.targets, self.domains = self._load_data()
#         self.num_samples = len(self.image_paths)
#         self.num_classes = len(set(self.targets))
#         self.num_domains = len(set(self.domains))
        
#         self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(set(self.targets)))}
#         self.domain_to_id = {domain: idx for idx, domain in enumerate(sorted(set(self.domains)))}
#         self.domain_to_indices = self._create_domain_indices()
        
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)
#         target = self.class_to_idx[self.targets[idx]]
#         domain = self.domain_to_id[self.domains[idx]]
        
#         return image, torch.tensor(target, dtype=torch.long), torch.tensor(domain, dtype=torch.long)

#     def _load_data(self) -> Tuple[List[str], List[str], List[str]]:
#         image_paths, targets, domains = [], [], []
#         for domain in os.listdir(self.dataset_path):
#             domain_path = os.path.join(self.dataset_path, domain)
#             if not os.path.isdir(domain_path):
#                 continue
#             for class_name in os.listdir(domain_path):
#                 class_path = os.path.join(domain_path, class_name)
#                 if not os.path.isdir(class_path):
#                     continue
#                 class_images = os.listdir(class_path)
#                 if self.max_samples_per_class:
#                     class_images = class_images[:self.max_samples_per_class]
#                 for image_name in class_images:
#                     image_paths.append(os.path.join(class_path, image_name))
#                     targets.append(class_name)
#                     domains.append(domain)
#         return image_paths, targets, domains

#     def create_incremental_split(self, n_domains: int) -> List[Tuple[Subset, Subset]]:
#         tasks = []
#         for domain_id in range(n_domains):
#             train_indices = [i for i, d in enumerate(self.domains) if self.domain_to_id[d] == domain_id]
#             test_indices = [i for i, d in enumerate(self.domains) if self.domain_to_id[d] != domain_id]
#             train_subset = Subset(self, train_indices)
#             test_subset = Subset(self, test_indices)
#             tasks.append((train_subset, test_subset))
#         return tasks

#     def create_data_loaders(self, tasks: List[Tuple[Subset, Subset]], batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader]]:
#         return [
#             (DataLoader(train_subset, batch_size=batch_size, shuffle=True),
#              DataLoader(test_subset, batch_size=batch_size, shuffle=False))
#             for train_subset, test_subset in tasks
#         ]

#     def __len__(self):
#         return self.num_samples

#     def print_samples_per_class_per_domain(self):
#         domain_class_counts = defaultdict(lambda: defaultdict(int))
#         for domain, target in zip(self.domains, self.targets):
#             domain_class_counts[domain][target] += 1
        
#         for domain, class_counts in domain_class_counts.items():
#             print(f"\nDomain: {domain}")
#             for class_name, count in class_counts.items():
#                 print(f"  Class {class_name}: {count} samples")
#             print(f"  Total samples: {sum(class_counts.values())}")

#     def _create_domain_indices(self):
#         domain_to_indices = {}
#         for idx, domain in enumerate(self.domains):
#             if domain not in domain_to_indices:
#                 domain_to_indices[domain] = []
#             domain_to_indices[domain].append(idx)
#         return domain_to_indices

#     def get_domain_data(self, domain):
#         if isinstance(domain, int):
#             domain = list(self.domain_to_indices.keys())[domain]
        
#         if domain not in self.domain_to_indices:
#             raise ValueError(f"Domain {domain} not found in the dataset.")
        
#         indices = self.domain_to_indices[domain]
#         return Subset(self, indices)