import torch
from .simple_attacks import create_attack

class AdversarialAttack:
    """
    A wrapper class for our custom adversarial attacks to match the advertorch API
    """
    def __init__(self, model, attack_type='fgsm', epsilon=0.03, clip_min=0.0, clip_max=1.0, **kwargs):
        """
        Initialize the adversarial attack
        
        Args:
            model: The model to attack
            attack_type: The type of attack to use
                ('fgsm', 'pgd', etc.)
            epsilon: The maximum perturbation size
            clip_min: Minimum value of the input
            clip_max: Maximum value of the input
            **kwargs: Additional attack-specific parameters
        """
        self.model = model
        self.attack_type = attack_type.lower()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = kwargs.get('targeted', False)
        
        # Map the attack types to our implementation
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
            # For CW, we just use PGD with more iterations as a simple substitute
            self.attack = create_attack('pgd', model, epsilon=epsilon, 
                                       alpha=kwargs.get('eps_iter', 0.005),
                                       num_iter=kwargs.get('nb_iter', 100),
                                       rand_init=kwargs.get('rand_init', True),
                                       clip_min=clip_min, clip_max=clip_max)
        else:
            raise ValueError(f"Attack type {attack_type} not supported in custom wrapper")
        
        # Set targeted attribute
        if hasattr(self.attack, 'targeted'):
            self.attack.targeted = self.targeted
    
    def generate(self, inputs, labels, targeted_labels=None):
        """
        Generate adversarial examples
        
        Args:
            inputs: Clean inputs
            labels: True labels for untargeted attacks, or target labels for targeted attacks
            targeted_labels: Target labels for targeted attacks
            
        Returns:
            perturbed_inputs: Adversarial examples
        """
        # For targeted attacks
        if targeted_labels is not None and self.targeted:
            return self.attack.perturb(inputs, targeted_labels)
        
        # For untargeted attacks
        return self.attack.perturb(inputs, labels)
    
    def perturb(self, inputs, labels):
        """
        Alias for generate to maintain compatibility with both APIs
        """
        return self.generate(inputs, labels)


class NoiseGenerator:
    """
    Class for generating different types of noise perturbations
    """
    def __init__(self, noise_type='gaussian', epsilon=0.03, clip_min=0.0, clip_max=1.0):
        """
        Initialize the noise generator
        
        Args:
            noise_type: Type of noise to generate
                ('gaussian', 'uniform', 'salt_and_pepper')
            epsilon: Magnitude of the noise
            clip_min: Minimum value of the input
            clip_max: Maximum value of the input
        """
        self.noise_type = noise_type.lower()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # Map the noise types to our implementation
        if self.noise_type == 'gaussian':
            self.noise_generator = create_attack('gaussian', None, epsilon=epsilon, 
                                                clip_min=clip_min, clip_max=clip_max)
        elif self.noise_type == 'uniform':
            self.noise_generator = create_attack('uniform', None, epsilon=epsilon, 
                                                clip_min=clip_min, clip_max=clip_max)
        elif self.noise_type == 'salt_and_pepper':
            # For salt and pepper, epsilon is interpreted as density
            self.noise_generator = create_attack('salt_and_pepper', None, density=epsilon, 
                                                clip_min=clip_min, clip_max=clip_max)
        else:
            raise ValueError(f"Noise type {noise_type} not supported")
    
    def generate(self, inputs):
        """
        Generate noisy examples
        
        Args:
            inputs: Clean inputs
            
        Returns:
            noisy_inputs: Perturbed inputs
        """
        return self.noise_generator.perturb(inputs)


class OcclusionGenerator:
    """
    Class for generating occlusion perturbations
    """
    def __init__(self, occlusion_type='block', occlusion_size=6, occlusion_value=0.0, random_position=True):
        """
        Initialize the occlusion generator
        
        Args:
            occlusion_type: Type of occlusion to generate
                ('block', 'random_blocks', 'gaussian_patch')
            occlusion_size: Size of the occlusion patch
            occlusion_value: Value to fill in the occluded region
            random_position: Whether to use random position for the occlusion
        """
        self.occlusion_type = occlusion_type.lower()
        self.occlusion_size = occlusion_size
        self.occlusion_value = occlusion_value
        self.random_position = random_position
        
        # For now, we only implement block occlusion
        self.occlusion_generator = create_attack('block', None, 
                                                block_size=occlusion_size, 
                                                value=occlusion_value,
                                                random_position=random_position)
    
    def generate(self, inputs):
        """
        Generate occluded examples
        
        Args:
            inputs: Clean inputs
            
        Returns:
            occluded_inputs: Inputs with occlusions
        """
        return self.occlusion_generator.perturb(inputs) 