import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import os


class DatasetLoader:
    def __init__(self, dataset_name, data_dir='./data', batch_size=128, num_workers=4, download=True):
        """
        Initialize dataset loader for image classification
        
        Args:
            dataset_name: Name of the dataset ('cifar10' or 'mnist')
            data_dir: Directory to store the dataset
            batch_size: Batch size for training and testing
            num_workers: Number of worker threads for data loading
            download: Whether to download the dataset if not available
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Set up transforms based on dataset
        if self.dataset_name == 'cifar10':
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
            
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            
            self.dataset_class = datasets.CIFAR10
            self.input_channels = 3
            self.num_classes = 10
            
        elif self.dataset_name == 'mnist':
            self.mean = (0.1307,)
            self.std = (0.3081,)
            
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            
            self.dataset_class = datasets.MNIST
            self.input_channels = 1
            self.num_classes = 10
            
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose 'cifar10' or 'mnist'.")
    
    def get_datasets(self, val_split=0.1):
        """
        Get training, validation, and test datasets
        
        Args:
            val_split: Fraction of training data to use for validation
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Load training data
        train_full = self.dataset_class(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.train_transform
        )
        
        # Load test data
        test_dataset = self.dataset_class(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.test_transform
        )
        
        # Split training data into training and validation sets
        val_size = int(len(train_full) * val_split)
        train_size = len(train_full) - val_size
        
        # Use random_split for reproducibility with a fixed generator
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            train_full, [train_size, val_size], generator=generator
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_loaders(self, val_split=0.1):
        """
        Get DataLoader objects for training, validation, and testing
        
        Args:
            val_split: Fraction of training data to use for validation
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset, val_dataset, test_dataset = self.get_datasets(val_split)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_single_loader(self, train=False, transform=None):
        """
        Get a DataLoader for either training or test data with optional custom transform
        
        Args:
            train: Whether to use training or test data
            transform: Optional custom transform
            
        Returns:
            data_loader
        """
        if transform is None:
            transform = self.train_transform if train else self.test_transform
            
        dataset = self.dataset_class(
            root=self.data_dir,
            train=train,
            download=self.download,
            transform=transform
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,  # Shuffle only if training
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return data_loader
    
    def get_dataset_info(self):
        """
        Get information about the dataset
        
        Returns:
            Dict containing dataset information
        """
        return {
            'name': self.dataset_name,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'mean': self.mean,
            'std': self.std
        } 