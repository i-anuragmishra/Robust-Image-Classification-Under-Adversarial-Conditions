import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
from tqdm import tqdm
from robustcnn.adversarial import AdversarialAttack, NoiseGenerator


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.001, weight_decay=1e-5,
                 optimizer_type='adam'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported. Choose 'adam' or 'sgd'")
        
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            
            loss.backward()
            self.optimizer.step()
            
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs=100, early_stopping_patience=10, scheduler=None):
        
        
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
            
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            
            if scheduler is not None:
                scheduler.step()
        
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history


class AdversarialTrainer(Trainer):
    
    def __init__(self, model, train_loader, val_loader, device, 
                 attack_type='fgsm', epsilon=0.03, 
                 learning_rate=0.001, weight_decay=1e-5, 
                 optimizer_type='adam', **attack_kwargs):
        
        super().__init__(model, train_loader, val_loader, device, learning_rate, weight_decay, optimizer_type)
        
        
        self.attack = AdversarialAttack(
            model, 
            attack_type=attack_type, 
            epsilon=epsilon, 
            clip_min=0.0, 
            clip_max=1.0, 
            **attack_kwargs
        )
    
    def train_epoch(self):
        
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Adversarial Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            
            perturbed_inputs = self.attack.generate(inputs, targets)
            
            
            self.optimizer.zero_grad()
            
            
            clean_outputs = self.model(inputs)
            adv_outputs = self.model(perturbed_inputs)
            
            
            clean_loss = self.criterion(clean_outputs, targets)
            adv_loss = self.criterion(adv_outputs, targets)
            loss = 0.5 * clean_loss + 0.5 * adv_loss
            
            
            loss.backward()
            self.optimizer.step()
            
            
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc


class DefensiveDistillationTrainer(Trainer):
    
    def __init__(self, model, train_loader, val_loader, device, 
                 teacher_model=None, temperature=20.0, alpha=0.5,
                 learning_rate=0.001, weight_decay=1e-5, 
                 optimizer_type='adam'):
        
        super().__init__(model, train_loader, val_loader, device, learning_rate, weight_decay, optimizer_type)
        
        
        self.teacher_model = teacher_model
        if self.teacher_model is None:
            raise ValueError("Teacher model must be provided for defensive distillation")
        
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
    
    def train_epoch(self):
        
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Distillation Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
                soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
            
            
            self.optimizer.zero_grad()
            student_outputs = self.model(inputs)
            student_soft = F.softmax(student_outputs / self.temperature, dim=1)
            
            
            hard_loss = self.criterion(student_outputs, targets)
            soft_loss = F.kl_div(torch.log(student_soft + 1e-8), soft_targets, reduction='batchmean')
            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.temperature ** 2)
            
            
            loss.backward()
            self.optimizer.step()
            
            
            train_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc


class NoiseAugmentationTrainer(Trainer):
    
    def __init__(self, model, train_loader, val_loader, device,
                 noise_type='gaussian', epsilon=0.03, 
                 learning_rate=0.001, weight_decay=1e-5, 
                 optimizer_type='adam'):
        
        super().__init__(model, train_loader, val_loader, device, learning_rate, weight_decay, optimizer_type)
        
        
        self.noise_generator = NoiseGenerator(
            noise_type=noise_type,
            epsilon=epsilon,
            clip_min=0.0,
            clip_max=1.0
        )
    
    def train_epoch(self):
        
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.train_loader, desc="Noise Augmentation Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            
            noisy_inputs = self.noise_generator.generate(inputs)
            
            
            self.optimizer.zero_grad()
            
            
            clean_outputs = self.model(inputs)
            noisy_outputs = self.model(noisy_inputs)
            
            
            clean_loss = self.criterion(clean_outputs, targets)
            noisy_loss = self.criterion(noisy_outputs, targets)
            loss = 0.5 * clean_loss + 0.5 * noisy_loss
            
            
            loss.backward()
            self.optimizer.step()
            
            
            train_loss += loss.item()
            
            
            _, predicted = clean_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc


def get_trainer(training_mode, model, train_loader, val_loader, device, **kwargs):
    
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