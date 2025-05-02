import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FGSM:
    
    def __init__(self, model, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = False
        self.loss_fn = nn.CrossEntropyLoss()
    
    def perturb(self, x, y):
        
        
        self.model.eval()
        
        
        x_adv = x.clone().detach().requires_grad_(True)
        
        
        outputs = self.model(x_adv)
        
        
        if self.targeted:
            loss = -self.loss_fn(outputs, y)  
        else:
            loss = self.loss_fn(outputs, y)  
        
        
        loss.backward()
        
        
        grad_sign = x_adv.grad.sign()
        
        
        if self.targeted:
            x_adv = x_adv - self.epsilon * grad_sign  
        else:
            x_adv = x_adv + self.epsilon * grad_sign  
        
        
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()


class PGD:
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_iter=20, 
                 rand_init=True, clip_min=0.0, clip_max=1.0):
        
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = False
        self.loss_fn = nn.CrossEntropyLoss()
    
    def perturb(self, x, y):
        
        
        self.model.eval()
        
        
        x_adv = x.clone().detach()
        
        
        if self.rand_init:
            x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        for _ in range(self.num_iter):
            
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            
            outputs = self.model(x_adv)
            
            
            if self.targeted:
                loss = -self.loss_fn(outputs, y)  
            else:
                loss = self.loss_fn(outputs, y)  
            
            
            loss.backward()
            
            
            if self.targeted:
                x_adv = x_adv - self.alpha * x_adv.grad.sign()  
            else:
                x_adv = x_adv + self.alpha * x_adv.grad.sign()  
            
            
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta
            
            
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()


class GaussianNoise:
    
    def __init__(self, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def perturb(self, x, y=None):
        
        noise = torch.randn_like(x) * self.epsilon
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, self.clip_min, self.clip_max)
        return x_noisy


class UniformNoise:
    
    def __init__(self, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def perturb(self, x, y=None):
        
        noise = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, self.clip_min, self.clip_max)
        return x_noisy


class SaltAndPepperNoise:
    
    def __init__(self, density=0.05, clip_min=0.0, clip_max=1.0):
        
        self.density = density
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def perturb(self, x, y=None):
        
        x_noisy = x.clone()
        
        
        salt_mask = torch.rand_like(x) < (self.density / 2)
        x_noisy[salt_mask] = self.clip_max
        
        
        pepper_mask = torch.rand_like(x) < (self.density / 2)
        x_noisy[pepper_mask] = self.clip_min
        
        return x_noisy


class BlockOcclusion:
    
    def __init__(self, block_size=5, value=0.0, random_position=True):
        
        self.block_size = block_size
        self.value = value
        self.random_position = random_position
    
    def perturb(self, x, y=None):
        
        x_occluded = x.clone()
        batch_size, channels, height, width = x.shape
        
        for i in range(batch_size):
            if self.random_position:
                
                h_start = np.random.randint(0, max(1, height - self.block_size))
                w_start = np.random.randint(0, max(1, width - self.block_size))
            else:
                
                h_start = (height - self.block_size) // 2
                w_start = (width - self.block_size) // 2
            
            h_end = min(height, h_start + self.block_size)
            w_end = min(width, w_start + self.block_size)
            
            
            x_occluded[i, :, h_start:h_end, w_start:w_end] = self.value
        
        return x_occluded



def create_attack(attack_type, model, **kwargs):
    
    if attack_type == 'fgsm':
        return FGSM(model, **kwargs)
    elif attack_type == 'pgd':
        return PGD(model, **kwargs)
    elif attack_type == 'gaussian':
        return GaussianNoise(**kwargs)
    elif attack_type == 'uniform':
        return UniformNoise(**kwargs)
    elif attack_type == 'salt_and_pepper':
        return SaltAndPepperNoise(**kwargs)
    elif attack_type == 'block':
        return BlockOcclusion(**kwargs)
    else:
        raise ValueError(f"Attack type {attack_type} not supported") 