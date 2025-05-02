import torch
from .simple_attacks import create_attack

class AdversarialAttack:
    
    def __init__(self, model, attack_type='fgsm', epsilon=0.03, clip_min=0.0, clip_max=1.0, **kwargs):
        
        self.model = model
        self.attack_type = attack_type.lower()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = kwargs.get('targeted', False)
        
        
        if self.attack_type == 'fgsm':
            self.attack = create_attack('fgsm', model, epsilon=epsilon, 
                                       clip_min=clip_min, clip_max=clip_max)
        elif self.attack_type in ['pgd', 'linf-pgd']:
            self.attack = create_attack('pgd', model, epsilon=epsilon, 
                                       alpha=kwargs.get('eps_iter', 0.01),
                                       num_iter=kwargs.get('nb_iter', 40),
                                       rand_init=kwargs.get('rand_init', True),
                                       clip_min=clip_min, clip_max=clip_max)
        elif self.attack_type == 'cw':
            
            self.attack = create_attack('pgd', model, epsilon=epsilon, 
                                       alpha=kwargs.get('eps_iter', 0.005),
                                       num_iter=kwargs.get('nb_iter', 100),
                                       rand_init=kwargs.get('rand_init', True),
                                       clip_min=clip_min, clip_max=clip_max)
        else:
            raise ValueError(f"Attack type {attack_type} not supported in custom wrapper")
        
        
        if hasattr(self.attack, 'targeted'):
            self.attack.targeted = self.targeted
    
    def generate(self, inputs, labels, targeted_labels=None):
        
        
        if targeted_labels is not None and self.targeted:
            return self.attack.perturb(inputs, targeted_labels)
        
        
        return self.attack.perturb(inputs, labels)
    
    def perturb(self, inputs, labels):
        
        return self.generate(inputs, labels)


class NoiseGenerator:
    
    def __init__(self, noise_type='gaussian', epsilon=0.03, clip_min=0.0, clip_max=1.0):
        
        self.noise_type = noise_type.lower()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        
        if self.noise_type == 'gaussian':
            self.noise_generator = create_attack('gaussian', None, epsilon=epsilon, 
                                                clip_min=clip_min, clip_max=clip_max)
        elif self.noise_type == 'uniform':
            self.noise_generator = create_attack('uniform', None, epsilon=epsilon, 
                                                clip_min=clip_min, clip_max=clip_max)
        elif self.noise_type == 'salt_and_pepper':
            
            self.noise_generator = create_attack('salt_and_pepper', None, density=epsilon, 
                                                clip_min=clip_min, clip_max=clip_max)
        else:
            raise ValueError(f"Noise type {noise_type} not supported")
    
    def generate(self, inputs):
        
        return self.noise_generator.perturb(inputs)


class OcclusionGenerator:
    
    def __init__(self, occlusion_type='block', occlusion_size=6, occlusion_value=0.0, random_position=True):
        
        self.occlusion_type = occlusion_type.lower()
        self.occlusion_size = occlusion_size
        self.occlusion_value = occlusion_value
        self.random_position = random_position
        
        
        self.occlusion_generator = create_attack('block', None, 
                                                block_size=occlusion_size, 
                                                value=occlusion_value,
                                                random_position=random_position)
    
    def generate(self, inputs):
        
        return self.occlusion_generator.perturb(inputs) 