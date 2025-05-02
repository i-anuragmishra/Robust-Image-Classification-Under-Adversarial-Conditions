import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
from tqdm import tqdm
from robustcnn.adversarial import AdversarialAttack, NoiseGenerator


class Trainer:
    """
    Base class for model training
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.001, weight_decay=1e-5,
                 optimizer_type='adam'):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer to use ('adam' or 'sgd')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set up optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported. Choose 'adam' or 'sgd'")
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        """
        Train the model for one epoch
        
        Returns:
            train_loss: Average training loss for the epoch
            train_acc: Training accuracy for the epoch
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            val_loss: Average validation loss
            val_acc: Validation accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs=100, early_stopping_patience=10, scheduler=None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for validation loss improvement
            scheduler: Learning rate scheduler
            
        Returns:
            history: Dictionary containing training history
        """
        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Step the scheduler if provided
            if scheduler is not None:
                scheduler.step()
        
        # Load the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history


class AdversarialTrainer(Trainer):
    """
    Trainer for adversarial training
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 attack_type='fgsm', epsilon=0.03, 
                 learning_rate=0.001, weight_decay=1e-5, 
                 optimizer_type='adam', **attack_kwargs):
        """
        Initialize the adversarial trainer
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training
            attack_type: Type of adversarial attack to use
            epsilon: Epsilon parameter for the attack
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer to use
            **attack_kwargs: Additional attack-specific parameters
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, weight_decay, optimizer_type)
        
        # Create adversarial attack
        self.attack = AdversarialAttack(
            model, 
            attack_type=attack_type, 
            epsilon=epsilon, 
            clip_min=0.0, 
            clip_max=1.0, 
            **attack_kwargs
        )
    
    def train_epoch(self):
        """
        Train the model for one epoch with adversarial examples
        
        Returns:
            train_loss: Average training loss for the epoch
            train_acc: Training accuracy for the epoch
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Adversarial Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Generate adversarial examples
            perturbed_inputs = self.attack.generate(inputs, targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Train on both clean and adversarial examples
            clean_outputs = self.model(inputs)
            adv_outputs = self.model(perturbed_inputs)
            
            # Calculate loss
            clean_loss = self.criterion(clean_outputs, targets)
            adv_loss = self.criterion(adv_outputs, targets)
            loss = 0.5 * clean_loss + 0.5 * adv_loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc


class DefensiveDistillationTrainer(Trainer):
    """
    Trainer for defensive distillation
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 teacher_model=None, temperature=20.0, alpha=0.5,
                 learning_rate=0.001, weight_decay=1e-5, 
                 optimizer_type='adam'):
        """
        Initialize the defensive distillation trainer
        
        Args:
            model: The student model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training
            teacher_model: Pretrained teacher model
            temperature: Temperature for softening the probabilities
            alpha: Weight for balancing hard and soft targets
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer to use
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, weight_decay, optimizer_type)
        
        # Set up teacher model
        self.teacher_model = teacher_model
        if self.teacher_model is None:
            raise ValueError("Teacher model must be provided for defensive distillation")
        
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
    
    def train_epoch(self):
        """
        Train the model for one epoch with defensive distillation
        
        Returns:
            train_loss: Average training loss for the epoch
            train_acc: Training accuracy for the epoch
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Distillation Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass with teacher model
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
                soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
            
            # Forward pass with student model
            self.optimizer.zero_grad()
            student_outputs = self.model(inputs)
            student_soft = F.softmax(student_outputs / self.temperature, dim=1)
            
            # Calculate loss
            hard_loss = self.criterion(student_outputs, targets)
            soft_loss = F.kl_div(torch.log(student_soft + 1e-8), soft_targets, reduction='batchmean')
            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.temperature ** 2)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc


class NoiseAugmentationTrainer(Trainer):
    """
    Trainer for noise augmentation
    """
    def __init__(self, model, train_loader, val_loader, device,
                 noise_type='gaussian', epsilon=0.03, 
                 learning_rate=0.001, weight_decay=1e-5, 
                 optimizer_type='adam'):
        """
        Initialize the noise augmentation trainer
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training
            noise_type: Type of noise to use for augmentation
            epsilon: Noise magnitude
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer to use
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, weight_decay, optimizer_type)
        
        # Create noise generator
        self.noise_generator = NoiseGenerator(
            noise_type=noise_type,
            epsilon=epsilon,
            clip_min=0.0,
            clip_max=1.0
        )
    
    def train_epoch(self):
        """
        Train the model for one epoch with noisy examples
        
        Returns:
            train_loss: Average training loss for the epoch
            train_acc: Training accuracy for the epoch
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Noise Augmentation Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Generate noisy examples
            noisy_inputs = self.noise_generator.generate(inputs)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Train on both clean and noisy examples
            clean_outputs = self.model(inputs)
            noisy_outputs = self.model(noisy_inputs)
            
            # Calculate loss
            clean_loss = self.criterion(clean_outputs, targets)
            noisy_loss = self.criterion(noisy_outputs, targets)
            loss = 0.5 * clean_loss + 0.5 * noisy_loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            
            # Measure accuracy on clean inputs
            _, predicted = clean_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc


def get_trainer(training_mode, model, train_loader, val_loader, device, **kwargs):
    """
    Factory function to get a trainer instance
    
    Args:
        training_mode: Type of training to use
            ('standard', 'adversarial', 'distillation', 'noise_augmentation')
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
        **kwargs: Additional trainer-specific parameters
        
    Returns:
        trainer: Instantiated trainer
    """
    if training_mode == 'standard':
        return Trainer(model, train_loader, val_loader, device, **kwargs)
    elif training_mode == 'adversarial':
        return AdversarialTrainer(model, train_loader, val_loader, device, **kwargs)
    elif training_mode == 'distillation':
        return DefensiveDistillationTrainer(model, train_loader, val_loader, device, **kwargs)
    elif training_mode == 'noise_augmentation':
        return NoiseAugmentationTrainer(model, train_loader, val_loader, device, **kwargs)
    else:
        raise ValueError(f"Training mode {training_mode} not supported") 