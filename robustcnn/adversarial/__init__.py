# Try to use our custom implementation first, if it fails fall back to advertorch
try:
    from .custom_wrapper import AdversarialAttack, NoiseGenerator, OcclusionGenerator
except ImportError:
    # Fall back to advertorch implementation
    try:
        from .attacks import AdversarialAttack, NoiseGenerator, OcclusionGenerator
    except ImportError:
        raise ImportError("Neither custom implementation nor advertorch implementation available")

__all__ = ['AdversarialAttack', 'NoiseGenerator', 'OcclusionGenerator'] 