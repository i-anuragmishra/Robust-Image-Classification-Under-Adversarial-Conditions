import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch.autograd.gradcheck
if not hasattr(torch.autograd.gradcheck, 'zero_gradients'):
    def zero_gradients(x):
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                zero_gradients(elem)
    
    
    torch.autograd.gradcheck.zero_gradients = zero_gradients

import collections.abc  

from advertorch.attacks import (
    GradientSignAttack,
    PGDAttack,
    CarliniWagnerL2Attack,
    LinfPGDAttack,
    JacobianSaliencyMapAttack
)


class AdversarialAttack:
    
    def __init__(self, model, attack_type='fgsm', epsilon=0.03, clip_min=0.0, clip_max=1.0, **kwargs):
        
        self.model = model
        self.attack_type = attack_type.lower()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        
        if self.attack_type == 'fgsm':
            self.attack = GradientSignAttack(
                self.model, 
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=self.epsilon,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                targeted=kwargs.get('targeted', False)
            )
        elif self.attack_type == 'pgd':
            self.attack = PGDAttack(
                self.model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=self.epsilon,
                nb_iter=kwargs.get('nb_iter', 40),
                eps_iter=kwargs.get('eps_iter', 0.01),
                rand_init=kwargs.get('rand_init', True),
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                targeted=kwargs.get('targeted', False)
            )
        elif self.attack_type == 'linf-pgd':
            self.attack = LinfPGDAttack(
                self.model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=self.epsilon,
                nb_iter=kwargs.get('nb_iter', 40),
                eps_iter=kwargs.get('eps_iter', 0.01),
                rand_init=kwargs.get('rand_init', True),
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                targeted=kwargs.get('targeted', False)
            )
        elif self.attack_type == 'cw':
            self.attack = CarliniWagnerL2Attack(
                self.model,
                kwargs.get('num_classes', 10),
                confidence=kwargs.get('confidence', 0),
                max_iterations=kwargs.get('max_iterations', 100),
                learning_rate=kwargs.get('learning_rate', 0.01),
                binary_search_steps=kwargs.get('binary_search_steps', 9),
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                targeted=kwargs.get('targeted', False)
            )
        elif self.attack_type == 'jsma':
            self.attack = JacobianSaliencyMapAttack(
                self.model,
                kwargs.get('num_classes', 10),
                gamma=kwargs.get('gamma', 1.0),
                theta=kwargs.get('theta', 1.0),
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                targeted=kwargs.get('targeted', True)
            )
        else:
            raise ValueError(f"Attack type {attack_type} not supported")
    
    def generate(self, inputs, labels, targeted_labels=None):
        
        
        if targeted_labels is not None and self.attack.targeted:
            return self.attack.perturb(inputs, targeted_labels)
        
        
        return self.attack.perturb(inputs, labels)


class NoiseGenerator:
    
    def __init__(self, noise_type='gaussian', epsilon=0.03, clip_min=0.0, clip_max=1.0):
        
        self.noise_type = noise_type.lower()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def generate(self, inputs):
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(inputs) * self.epsilon
            noisy_inputs = inputs + noise
        
        elif self.noise_type == 'uniform':
            noise = torch.rand_like(inputs) * 2 * self.epsilon - self.epsilon
            noisy_inputs = inputs + noise
        
        elif self.noise_type == 'salt_and_pepper':
            noise = torch.zeros_like(inputs)
            salt = torch.bernoulli(torch.ones_like(inputs) * 0.5) * 2 * self.epsilon
            pepper = -torch.bernoulli(torch.ones_like(inputs) * 0.5) * 2 * self.epsilon
            noise = salt + pepper
            noisy_inputs = inputs + noise
        
        else:
            raise ValueError(f"Noise type {self.noise_type} not supported")
        
        
        noisy_inputs = torch.clamp(noisy_inputs, self.clip_min, self.clip_max)
        
        return noisy_inputs


class OcclusionGenerator:
    
    def __init__(self, occlusion_type='block', occlusion_size=6, occlusion_value=0.0, random_position=True):
        
        self.occlusion_type = occlusion_type.lower()
        self.occlusion_size = occlusion_size
        self.occlusion_value = occlusion_value
        self.random_position = random_position
    
    def generate(self, inputs):
        
        batch_size, channels, height, width = inputs.shape
        occluded_inputs = inputs.clone()
        
        if self.occlusion_type == 'block':
            for i in range(batch_size):
                if self.random_position:
                    
                    h_start = np.random.randint(0, height - self.occlusion_size + 1)
                    w_start = np.random.randint(0, width - self.occlusion_size + 1)
                else:
                    
                    h_start = (height - self.occlusion_size) // 2
                    w_start = (width - self.occlusion_size) // 2
                
                occluded_inputs[i, :, h_start:h_start+self.occlusion_size, w_start:w_start+self.occlusion_size] = self.occlusion_value
        
        elif self.occlusion_type == 'random_blocks':
            
            num_blocks = 5
            block_size = self.occlusion_size // 2
            
            for i in range(batch_size):
                for _ in range(num_blocks):
                    h_start = np.random.randint(0, height - block_size + 1)
                    w_start = np.random.randint(0, width - block_size + 1)
                    occluded_inputs[i, :, h_start:h_start+block_size, w_start:w_start+block_size] = self.occlusion_value
        
        elif self.occlusion_type == 'gaussian_patch':
            
            x = np.linspace(-1, 1, self.occlusion_size)
            y = np.linspace(-1, 1, self.occlusion_size)
            x_grid, y_grid = np.meshgrid(x, y)
            z = np.exp(-(x_grid**2 + y_grid**2) / 0.5)
            mask = torch.from_numpy(z).float().to(inputs.device)
            
            for i in range(batch_size):
                if self.random_position:
                    h_start = np.random.randint(0, height - self.occlusion_size + 1)
                    w_start = np.random.randint(0, width - self.occlusion_size + 1)
                else:
                    h_start = (height - self.occlusion_size) // 2
                    w_start = (width - self.occlusion_size) // 2
                
                for c in range(channels):
                    patch = occluded_inputs[i, c, h_start:h_start+self.occlusion_size, w_start:w_start+self.occlusion_size]
                    occluded_inputs[i, c, h_start:h_start+self.occlusion_size, w_start:w_start+self.occlusion_size] = patch * (1 - mask) + self.occlusion_value * mask
        
        else:
            raise ValueError(f"Occlusion type {self.occlusion_type} not supported")
        
        return occluded_inputs 