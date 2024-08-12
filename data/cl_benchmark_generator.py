import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict
from collections import defaultdict

# src/data/cl_benchmark_generator.py



class CLBenchmarkGenerator(Dataset):
    """
    Custom dataset class for benchmarking continual learning models.
    """

    def __init__(self, dataset_path: str, max_samples_per_class: int = None, augmentation: str = None):
        """
        Initialize the CLBenchmarkGenerator dataset.

        Args:
            dataset_path (str): Path to the dataset directory.
            max_samples_per_class (int, optional): Maximum number of samples per class. Defaults to None.
            augmentation (str, optional): Augmentation techniques to apply. Defaults to None.
        """
        self.dataset_path = dataset_path
        self.max_samples_per_class = max_samples_per_class
        self.image_paths, self.targets, self.domains = self._load_data()
        self.num_samples = len(self.image_paths)
        self.num_domains = len(set(self.domains))
        
        unique_classes = sorted(set(self.targets))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.num_classes = len(self.class_to_idx)
        self.domain_to_indices = self._create_domain_indices()
        self.transform = self._get_transform(augmentation)

    def _get_transform(self, augmentation: str = None):
        """
        Get the data transformation pipeline.

        Args:
            augmentation (str, optional): Augmentation techniques to apply. Defaults to None.

        Returns:
            torchvision.transforms.Compose: Composed transformation pipeline.
        """
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if augmentation is None:
            return base_transform

        augmentation_transforms = []
        if 'random_crop' in augmentation:
            augmentation_transforms.append(transforms.RandomCrop(224, padding=4))
        if 'random_flip' in augmentation:
            augmentation_transforms.append(transforms.RandomHorizontalFlip())
        if 'color_jitter' in augmentation:
            augmentation_transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if 'rotation' in augmentation:
            augmentation_transforms.append(transforms.RandomRotation(15))
        if 'grayscale' in augmentation:
            augmentation_transforms.append(transforms.RandomGrayscale(p=0.2))
        if 'gaussian_blur' in augmentation:
            augmentation_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

        return transforms.Compose(augmentation_transforms + [base_transform])

    def _load_data(self):
        """
        Load the dataset from the given path.

        Returns:
            Tuple[List[str], List[str], List[str]]: Image paths, targets, and domains.
        """
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
        """
        Get the item at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: Image, target, and domain.
        """
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
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.num_samples

    def update_class_indices(self, global_class_to_idx):
        """
        Update the class indices.

        Args:
            global_class_to_idx (Dict[str, int]): Global class to index mapping.
        """
        self.class_to_idx = global_class_to_idx
        self.num_classes = len(self.class_to_idx)

    def create_incremental_split(self, n_domains: int) -> List[Tuple[Subset, Subset]]:
        """
        Create incremental splits for the given number of domains.

        Args:
            n_domains (int): Number of domains.

        Returns:
            List[Tuple[Subset, Subset]]: List of train-test splits for each domain.
        """
        tasks = []
        for domain_id in range(n_domains):
            train_indices = [i for i, d in enumerate(self.domains) if self.domain_to_id[d] == domain_id]
            test_indices = [i for i, d in enumerate(self.domains) if self.domain_to_id[d] != domain_id]
            train_subset = Subset(self, train_indices)
            test_subset = Subset(self, test_indices)
            tasks.append((train_subset, test_subset))
        return tasks
    
    def get_domain_data(self, domain_id):
        """
        Get the data for a specific domain.

        Args:
            domain_id (int): Domain ID.

        Returns:
            Subset: Subset of the dataset for the specified domain.
        """
        domain_indices = [i for i, d in enumerate(self.domains) if d == domain_id]
        return Subset(self, domain_indices)

    def create_data_loaders(self, tasks: List[Tuple[Subset, Subset]], batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create data loaders for the given train-test splits.

        Args:
            tasks (List[Tuple[Subset, Subset]]): List of train-test splits.
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            List[Tuple[DataLoader, DataLoader]]: List of train-test data loaders.
        """
        return [
            (DataLoader(train_subset, batch_size=batch_size, shuffle=True),
             DataLoader(test_subset, batch_size=batch_size, shuffle=False))
            for train_subset, test_subset in tasks
        ]


    def print_samples_per_class_per_domain(self):
        """
        Print the number of samples per class per domain.
        """
        domain_class_counts = defaultdict(lambda: defaultdict(int))
        for domain, target in zip(self.domains, self.targets):
            domain_class_counts[domain][target] += 1
        
        for domain, class_counts in domain_class_counts.items():
            print(f"\nDomain: {domain}")
            for class_name, count in class_counts.items():
                print(f"  Class {class_name}: {count} samples")
            print(f"  Total samples: {sum(class_counts.values())}")

    def _create_domain_indices(self):
        """
        Create a mapping of domain to indices.

        Returns:
            Dict[str, List[int]]: Mapping of domain to indices.
        """
        domain_to_indices = {}
        for idx, domain in enumerate(self.domains):
            if domain not in domain_to_indices:
                domain_to_indices[domain] = []
            domain_to_indices[domain].append(idx)
        return domain_to_indices

    def get_domain_data(self, domain):
        """
        Get the data for a specific domain.

        Args:
            domain (Union[int, str]): Domain ID or domain name.

        Returns:
            Subset: Subset of the dataset for the specified domain.
        """
        if isinstance(domain, int):
            domain = list(self.domain_to_indices.keys())[domain]
        
        if domain not in self.domain_to_indices:
            raise ValueError(f"Domain {domain} not found in the dataset.")
        
        indices = self.domain_to_indices[domain]
        return Subset(self, indices)
