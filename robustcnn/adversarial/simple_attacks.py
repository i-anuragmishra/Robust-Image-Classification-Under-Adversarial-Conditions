import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FGSM:
    """
    Implementation of Fast Gradient Sign Method (FGSM)
    Paper: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, model, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        """
        Initialize FGSM attack
        
        Args:
            model: Model to attack
            epsilon: Attack strength
            clip_min: Minimum value of input
            clip_max: Maximum value of input
        """
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = False
        self.loss_fn = nn.CrossEntropyLoss()
    
    def perturb(self, x, y):
        """
        Generate adversarial examples
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            x_adv: Adversarial examples
        """
        # Make sure model is in evaluation mode
        self.model.eval()
        
        # Create a copy of the input that requires gradients
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        
        # Calculate loss
        if self.targeted:
            loss = -self.loss_fn(outputs, y)  # Target is desired class (minimize negative loss)
        else:
            loss = self.loss_fn(outputs, y)  # Target is true class (maximize loss)
        
        # Backward pass
        loss.backward()
        
        # Get gradient sign
        grad_sign = x_adv.grad.sign()
        
        # Create adversarial example
        if self.targeted:
            x_adv = x_adv - self.epsilon * grad_sign  # Move away from target
        else:
            x_adv = x_adv + self.epsilon * grad_sign  # Move toward maximum loss
        
        # Clamp to ensure valid range
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()


class PGD:
    """
    Implementation of Projected Gradient Descent (PGD)
    Paper: https://arxiv.org/abs/1706.06083
    """
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_iter=20, 
                 rand_init=True, clip_min=0.0, clip_max=1.0):
        """
        Initialize PGD attack
        
        Args:
            model: Model to attack
            epsilon: Attack strength (max norm of perturbation)
            alpha: Step size for each iteration
            num_iter: Number of iterations
            rand_init: Whether to use random initialization
            clip_min: Minimum value of input
            clip_max: Maximum value of input
        """
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
        """
        Generate adversarial examples
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            x_adv: Adversarial examples
        """
        # Make sure model is in evaluation mode
        self.model.eval()
        
        # Initialize adversarial examples
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.rand_init:
            x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            # Clamp to ensure valid range
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        for _ in range(self.num_iter):
            # Create a copy that requires gradients
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # Calculate loss
            if self.targeted:
                loss = -self.loss_fn(outputs, y)  # Target is desired class (minimize negative loss)
            else:
                loss = self.loss_fn(outputs, y)  # Target is true class (maximize loss)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial examples
            if self.targeted:
                x_adv = x_adv - self.alpha * x_adv.grad.sign()  # Move away from target
            else:
                x_adv = x_adv + self.alpha * x_adv.grad.sign()  # Move toward maximum loss
            
            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + delta
            
            # Clamp to ensure valid range
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        
        return x_adv.detach()


class GaussianNoise:
    """
    Add Gaussian noise to images
    """
    def __init__(self, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        """
        Initialize Gaussian noise
        
        Args:
            epsilon: Strength of noise
            clip_min: Minimum value of input
            clip_max: Maximum value of input
        """
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def perturb(self, x, y=None):
        """
        Add Gaussian noise to images
        
        Args:
            x: Input images
            y: Not used (kept for API consistency)
            
        Returns:
            x_noisy: Noisy images
        """
        noise = torch.randn_like(x) * self.epsilon
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, self.clip_min, self.clip_max)
        return x_noisy


class UniformNoise:
    """
    Add uniform noise to images
    """
    def __init__(self, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        """
        Initialize uniform noise
        
        Args:
            epsilon: Strength of noise
            clip_min: Minimum value of input
            clip_max: Maximum value of input
        """
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def perturb(self, x, y=None):
        """
        Add uniform noise to images
        
        Args:
            x: Input images
            y: Not used (kept for API consistency)
            
        Returns:
            x_noisy: Noisy images
        """
        noise = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, self.clip_min, self.clip_max)
        return x_noisy


class SaltAndPepperNoise:
    """
    Add salt and pepper noise to images
    """
    def __init__(self, density=0.05, clip_min=0.0, clip_max=1.0):
        """
        Initialize salt and pepper noise
        
        Args:
            density: Proportion of pixels to change
            clip_min: Minimum value of input
            clip_max: Maximum value of input
        """
        self.density = density
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def perturb(self, x, y=None):
        """
        Add salt and pepper noise to images
        
        Args:
            x: Input images
            y: Not used (kept for API consistency)
            
        Returns:
            x_noisy: Noisy images
        """
        x_noisy = x.clone()
        
        # Create salt noise (white pixels)
        salt_mask = torch.rand_like(x) < (self.density / 2)
        x_noisy[salt_mask] = self.clip_max
        
        # Create pepper noise (black pixels)
        pepper_mask = torch.rand_like(x) < (self.density / 2)
        x_noisy[pepper_mask] = self.clip_min
        
        return x_noisy


class BlockOcclusion:
    """
    Add block occlusion to images
    """
    def __init__(self, block_size=5, value=0.0, random_position=True):
        """
        Initialize block occlusion
        
        Args:
            block_size: Size of occlusion block
            value: Value to fill in the occluded region
            random_position: Whether to use random position
        """
        self.block_size = block_size
        self.value = value
        self.random_position = random_position
    
    def perturb(self, x, y=None):
        """
        Add block occlusion to images
        
        Args:
            x: Input images
            y: Not used (kept for API consistency)
            
        Returns:
            x_occluded: Occluded images
        """
        x_occluded = x.clone()
        batch_size, channels, height, width = x.shape
        
        for i in range(batch_size):
            if self.random_position:
                # Random position
                h_start = np.random.randint(0, max(1, height - self.block_size))
                w_start = np.random.randint(0, max(1, width - self.block_size))
            else:
                # Center position
                h_start = (height - self.block_size) // 2
                w_start = (width - self.block_size) // 2
            
            h_end = min(height, h_start + self.block_size)
            w_end = min(width, w_start + self.block_size)
            
            # Add occlusion
            x_occluded[i, :, h_start:h_end, w_start:w_end] = self.value
        
        return x_occluded


# Factory function to create attacks
def create_attack(attack_type, model, **kwargs):
    """
    Create an attack instance
    
    Args:
        attack_type: Type of attack to create
        model: Model to attack
        **kwargs: Attack-specific parameters
        
    Returns:
        attack: Attack instance
    """
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