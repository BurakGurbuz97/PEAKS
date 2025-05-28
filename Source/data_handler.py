from typing import Tuple, Callable, List, Iterator
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.datasets import CIFAR100, Food101
import random
from Source.custom_datasets import Food101N, WebVision
import math
import copy

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Using this is at least 30% than naively indexing the dataset 
class DynamicBatchSampler:
    """A sampler that can be updated with new indices and splits them into sub-batches for parallel processing."""
    def __init__(self, batch_size: int = 16):
        """
        Args:
            batch_size: The size of sub-batches to create for parallel processing
        """
        self.indices: List[int] = []
        self.batch_size = batch_size
    
    def update_indices(self, indices: List[int]) -> None:
        """Update the indices to be sampled."""
        self.indices = indices

    def __iter__(self) -> Iterator[List[int]]:
        """Yield sub-batches of indices for parallel processing.
        
        This creates smaller batches that can be distributed across workers,
        while ensuring all indices are processed.
        """
        total_samples = len(self.indices)
        
        # Calculate number of batches
        num_batches = math.ceil(total_samples / self.batch_size)
        
        # Yield batches of indices
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            yield self.indices[start_idx:end_idx]
        
    def __len__(self) -> int:
        """Return the number of batches."""
        return math.ceil(len(self.indices) / self.batch_size)


class IndexedDataset(Dataset):
    def __init__(self, subset, original_indices):
        self.subset = subset
        self.original_indices = original_indices

    def __getitem__(self, idx):
        sample, label = self.subset[idx]
        index = self.original_indices[idx]
        return index, sample, label

    def __len__(self):
        return len(self.subset)
        
class DataPoolManager:
    """Manages the datasets T_s, S, and D \ T_s for incremental data selection."""

    def __init__(self, full_dataset: Dataset, args) -> None:
        self.full_dataset = full_dataset
        self.args = args
        self.all_indices = set(range(len(full_dataset)))
        self.T_s_indices = set()  # Training set T_s
        self.S_indices = set()    # Temporary buffer S
        self.pool_indices = self.all_indices - self.T_s_indices  # D \ T_s
        self.random = random.Random(args.seed + 2000000)  # Add a large number to seed to avoid overlap with other seeds

        # Array of class counts in T_s
        self.class_counts = [0] * len(full_dataset.classes)

        # For termination condition
        self.k = args.k

        # Initialize the reusable DataLoader components
        self.batch_sampler = DynamicBatchSampler(batch_size=16)
        self.loader = DataLoader(
            dataset=self.full_dataset,
            batch_sampler=self.batch_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=None,
        )
        self.loader_iter = None

    def get_dataloader(self, indices: List[int], batch_size: int, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the specified indices.

        Args:
            indices: List of indices to include in the DataLoader.
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader instance.
        """
        subset = self.get_dataset_from_indices(indices)
        indexed_subset = IndexedDataset(subset, indices)
        data_loader = torch.utils.data.DataLoader(indexed_subset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def get_data(self, indices: List[int]) -> List[Tuple[torch.Tensor, int]]:
        """Retrieve data (inputs and labels) for the given indices.

        Args:
            indices: List of indices to retrieve data for.

        Returns:
            List of tuples containing (input_tensor, label).
        """
        self.batch_sampler.update_indices(indices)
        # Reset the iterator when indices change
        self.loader_iter = iter(self.loader)
        
        # Collect all sub-batches
        data = []
        for _ in range(len(self.batch_sampler)):
            batch = next(self.loader_iter)
            batch_data = list(zip(*batch))
            data.extend(batch_data)
        return data
    
    def __del__(self):
        """Cleanup DataLoader workers when the manager is destroyed."""
        if hasattr(self, 'loader'):
            del self.loader._iterator

    def initialize_T_s(self) -> None:
        """Initialize T_s with m random examples."""
        initial_indices = self.random.sample(self.pool_indices, self.args.initial_training_size)
        self.T_s_indices.update(initial_indices)
        self.pool_indices -= self.T_s_indices

        # update class counts
        for idx in initial_indices:
            _, label = self.full_dataset[idx]
            self.class_counts[label] += 1

    def add_to_S(self, indices: List[int]) -> None:
        """Add selected indices to S and remove from pool."""
        self.S_indices.update(indices)
        self.pool_indices -= set(indices)

        # update class counts
        for idx in indices:
            _, label = self.full_dataset[idx]
            self.class_counts[label] += 1

    def move_S_to_T_s(self) -> None:
        """Move all examples from S to T_s."""
        self.T_s_indices.update(self.S_indices)
        self.S_indices.clear()

    def sample_from_T_s(self, num_samples: int) -> List[int]:
        """Sample num_samples indices from T_s."""
        return self.random.sample(self.T_s_indices, min(num_samples, len(self.T_s_indices)))

    def get_dataset_from_indices(self, indices: List[int]) -> Dataset:
        """Return a Subset of full_dataset with specified indices."""
        return CustomSubset(self.full_dataset, indices)

    def get_dataset_T_s(self) -> Dataset:
        """Return the dataset for T_s."""
        return self.get_dataset_from_indices(list(self.T_s_indices))

    def get_dataset_T_end(self) -> Dataset:
        """Return the final training set T_end = T_s U S."""
        T_end_indices = self.T_s_indices.union(self.S_indices)
        return self.get_dataset_from_indices(list(T_end_indices))
    
    def get_T_s_indices(self) -> List[int]:
        """Return the list of indices in T_s."""
        return list(self.T_s_indices)

    def termination_condition_met(self, delta: int) -> bool:
        """Check termination condition."""
        is_done = len(self.T_s_indices) + len(self.S_indices) + delta >= self.k

        if is_done:
            self.move_S_to_T_s()
            if len(self.T_s_indices) < self.k:
                self.add_random_examples(self.k - len(self.T_s_indices))
                print(f"Added {self.k - len(self.T_s_indices)} random examples to T_s.")

        return is_done

    def get_pool_size(self) -> int:
        """Return the size of the pool D \ T_s."""
        return len(self.pool_indices)
    
    def get_total_selected_examples(self) -> int:
        """Return the total number of examples selected so far (size of T_s + S)."""
        return len(self.T_s_indices) + len(self.S_indices)
    
    def add_random_examples(self, num_examples: int) -> None:
        # Ensure we do not request more samples than are available in the pool
        num_samples_to_add = min(num_examples, len(self.pool_indices))
        if num_samples_to_add == 0:
            print("No samples available in the pool to add to T_s.")
            return
        random_indices = self.random.sample(self.pool_indices, num_samples_to_add)
        self.T_s_indices.update(random_indices)
        self.pool_indices -= set(random_indices)


class CustomSubset(Dataset):
    """Custom subset class that preserves dataset attributes including classes."""

    def __init__(self, dataset: Dataset, indices: List[int], transform=None) -> None:
        """
        Args:
            dataset: Original dataset
            indices: List of indices to subsample
        """
        self.dataset = copy.deepcopy(dataset)
        self.indices = indices
        self.transform = transform

        if transform and hasattr(self.dataset, 'transform'):
            self.dataset.transform  = None


        # Explicitly handle classes attribute
        if hasattr(dataset, 'classes'):
            self.classes = dataset.classes
        elif hasattr(dataset, 'class_to_idx'):
            self.classes = list(dataset.class_to_idx.keys())
        else:
            raise AttributeError(
                "Dataset has no 'classes' or 'class_to_idx' attribute"
            )

        # Preserve other important attributes
        for attr in dir(dataset):
            if not attr.startswith('__') and not callable(getattr(dataset, attr)):
                if not hasattr(self, attr):  # Don't override already set attributes
                    setattr(self, attr, getattr(dataset, attr))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item with index from subset."""
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        """Return length of subset."""
        return len(self.indices)



def get_cifar_transforms() -> Tuple[Callable, Callable]:
    common_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    return common_transforms, common_transforms


def get_general_transforms() -> Tuple[Callable, Callable]:
    common_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    return common_transforms, common_transforms



def get_dataset_and_transforms(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset, Tuple[Callable, Callable]]:
    """Get dataset and transforms based on dataset name.

    Returns:
        full_dataset, test_dataset, (train_transforms, test_transforms)

    Dataset output format:
        images: torch.Tensor[3, 224, 224]
        labels: torch.Tensor[]
    """
    if dataset_name == 'cifar100':
        train_transforms, test_transforms = get_cifar_transforms()
        train_dataset = CIFAR100(
            root='../data', train=True, download=True,
            transform=train_transforms
        )
        test_dataset = CIFAR100(
            root='../data', train=False, download=True,
            transform=test_transforms
        )
    elif dataset_name == 'food101':
        train_transforms, test_transforms = get_general_transforms()
        train_dataset = Food101(
            root='../data', split='train', download=True,
            transform=train_transforms
        )
        test_dataset = Food101(
            root='../data', split='test', download=True,
            transform=test_transforms
        )
    elif dataset_name == 'food101-noise':
        train_transforms, test_transforms = get_general_transforms()
        train_dataset = Food101N(
            root='../data', transform=train_transforms
        )
        test_dataset = Food101(
            root='../data', split='test', download=True,
            transform=test_transforms
        )
    elif dataset_name == 'webvision':
        train_transforms, test_transforms = get_general_transforms()
        train_dataset = WebVision(
            root='../data', split = "train", transform=train_transforms
        )
        test_dataset = WebVision(
            root='../data', split = "val", transform=test_transforms
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, (train_transforms, test_transforms)