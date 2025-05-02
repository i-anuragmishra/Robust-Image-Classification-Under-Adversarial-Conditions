import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import os
from tqdm import tqdm
from robustcnn.adversarial import AdversarialAttack, NoiseGenerator, OcclusionGenerator


def evaluate_model(model, data_loader, device):
    """
    Evaluate a model on a dataset
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        
    Returns:
        accuracy: Classification accuracy
        predictions: Model predictions
        targets: True labels
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Track metrics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, np.array(all_predictions), np.array(all_targets)


def evaluate_under_attack(model, data_loader, device, attack_type='fgsm', epsilon=0.03, **attack_kwargs):
    """
    Evaluate a model under adversarial attack
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        attack_type: Type of adversarial attack to use
        epsilon: Perturbation magnitude
        **attack_kwargs: Additional attack-specific parameters
        
    Returns:
        accuracy: Classification accuracy under attack
        clean_accuracy: Classification accuracy on clean data
        predictions: Model predictions under attack
        targets: True labels
    """
    model.eval()
    correct = 0
    clean_correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Create adversarial attack
    attack = AdversarialAttack(
        model,
        attack_type=attack_type,
        epsilon=epsilon,
        clip_min=0.0,
        clip_max=1.0,
        **attack_kwargs
    )
    
    for inputs, targets in tqdm(data_loader, desc=f"Evaluating under {attack_type.upper()} attack"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Evaluate on clean data
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(targets).sum().item()
        
        # Generate adversarial examples
        perturbed_inputs = attack.generate(inputs, targets)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(perturbed_inputs)
            _, predicted = outputs.max(1)
            
            # Track metrics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    clean_accuracy = 100. * clean_correct / total
    return accuracy, clean_accuracy, np.array(all_predictions), np.array(all_targets)


def evaluate_under_noise(model, data_loader, device, noise_type='gaussian', epsilon=0.03):
    """
    Evaluate a model under noise perturbations
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        noise_type: Type of noise to use
        epsilon: Noise magnitude
        
    Returns:
        accuracy: Classification accuracy under noise
        clean_accuracy: Classification accuracy on clean data
        predictions: Model predictions under noise
        targets: True labels
    """
    model.eval()
    correct = 0
    clean_correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Create noise generator
    noise_generator = NoiseGenerator(
        noise_type=noise_type,
        epsilon=epsilon,
        clip_min=0.0,
        clip_max=1.0
    )
    
    for inputs, targets in tqdm(data_loader, desc=f"Evaluating under {noise_type} noise"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Evaluate on clean data
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(targets).sum().item()
        
        # Generate noisy examples
        noisy_inputs = noise_generator.generate(inputs)
        
        # Evaluate on noisy examples
        with torch.no_grad():
            outputs = model(noisy_inputs)
            _, predicted = outputs.max(1)
            
            # Track metrics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    clean_accuracy = 100. * clean_correct / total
    return accuracy, clean_accuracy, np.array(all_predictions), np.array(all_targets)


def evaluate_under_occlusion(model, data_loader, device, occlusion_type='block', occlusion_size=6, occlusion_value=0.0):
    """
    Evaluate a model under occlusion perturbations
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        occlusion_type: Type of occlusion to use
        occlusion_size: Size of the occlusion
        occlusion_value: Value to fill in the occluded region
        
    Returns:
        accuracy: Classification accuracy under occlusion
        clean_accuracy: Classification accuracy on clean data
        predictions: Model predictions under occlusion
        targets: True labels
    """
    model.eval()
    correct = 0
    clean_correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Create occlusion generator
    occlusion_generator = OcclusionGenerator(
        occlusion_type=occlusion_type,
        occlusion_size=occlusion_size,
        occlusion_value=occlusion_value
    )
    
    for inputs, targets in tqdm(data_loader, desc=f"Evaluating under {occlusion_type} occlusion"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Evaluate on clean data
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(targets).sum().item()
        
        # Generate occluded examples
        occluded_inputs = occlusion_generator.generate(inputs)
        
        # Evaluate on occluded examples
        with torch.no_grad():
            outputs = model(occluded_inputs)
            _, predicted = outputs.max(1)
            
            # Track metrics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    clean_accuracy = 100. * clean_correct / total
    return accuracy, clean_accuracy, np.array(all_predictions), np.array(all_targets)


def plot_confusion_matrix(targets, predictions, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        targets: True labels
        predictions: Model predictions
        class_names: Names of the classes
        save_path: Path to save the plot
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(model, data_loader, device, num_classes, save_path=None):
    """
    Plot ROC curve for multi-class classification
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        num_classes: Number of classes
        save_path: Path to save the plot
    """
    model.eval()
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Computing ROC curve"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)
            
            all_targets.extend(targets.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)
    
    # Create one-hot encoding of targets
    targets_one_hot = np.zeros((all_targets.size, num_classes))
    targets_one_hot[np.arange(all_targets.size), all_targets] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], all_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_adversarial_examples(model, data_loader, device, attack_type='fgsm', epsilon=0.03, 
                                  num_examples=5, save_path=None, **attack_kwargs):
    """
    Visualize adversarial examples
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        attack_type: Type of adversarial attack to use
        epsilon: Perturbation magnitude
        num_examples: Number of examples to visualize
        save_path: Path to save the plot
        **attack_kwargs: Additional attack-specific parameters
    """
    model.eval()
    
    # Create adversarial attack
    attack = AdversarialAttack(
        model,
        attack_type=attack_type,
        epsilon=epsilon,
        clip_min=0.0,
        clip_max=1.0,
        **attack_kwargs
    )
    
    # Get examples
    examples = []
    count = 0
    
    for inputs, targets in data_loader:
        if count >= num_examples:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        perturbed_inputs = attack.generate(inputs, targets)
        
        # Get predictions
        with torch.no_grad():
            clean_outputs = model(inputs)
            adv_outputs = model(perturbed_inputs)
            
            _, clean_predicted = clean_outputs.max(1)
            _, adv_predicted = adv_outputs.max(1)
        
        # Find misclassified examples
        for i in range(inputs.size(0)):
            if count >= num_examples:
                break
                
            if clean_predicted[i] == targets[i] and adv_predicted[i] != targets[i]:
                examples.append({
                    'clean': inputs[i].cpu(),
                    'adversarial': perturbed_inputs[i].cpu(),
                    'perturbation': (perturbed_inputs[i] - inputs[i]).cpu(),
                    'target': targets[i].item(),
                    'clean_pred': clean_predicted[i].item(),
                    'adv_pred': adv_predicted[i].item()
                })
                count += 1
    
    # Visualize examples
    if examples:
        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 2 * num_examples))
        
        for i, example in enumerate(examples):
            # Unnormalize images for visualization
            clean_img = example['clean'].permute(1, 2, 0).numpy()
            adv_img = example['adversarial'].permute(1, 2, 0).numpy()
            pert_img = example['perturbation'].permute(1, 2, 0).numpy()
            
            # Scale perturbation for visibility
            pert_img = (pert_img - pert_img.min()) / (pert_img.max() - pert_img.min() + 1e-8)
            
            # Plot
            axes[i, 0].imshow(clean_img)
            axes[i, 0].set_title(f"Original (Pred: {example['clean_pred']}, True: {example['target']})")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(adv_img)
            axes[i, 1].set_title(f"Adversarial (Pred: {example['adv_pred']})")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pert_img)
            axes[i, 2].set_title("Perturbation (Magnified)")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    else:
        print("No suitable adversarial examples found. Try with more examples or a different epsilon.")


def evaluate_model_robustness(model, data_loader, device, attack_types=['fgsm', 'pgd'], 
                             epsilons=[0.01, 0.03, 0.05, 0.1], 
                             noise_types=['gaussian', 'uniform'], 
                             occlusion_params=[{'type': 'block', 'size': 4}, {'type': 'block', 'size': 8}],
                             results_dir='./results'):
    """
    Comprehensive evaluation of model robustness
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        attack_types: Types of adversarial attacks to evaluate
        epsilons: Perturbation magnitudes to evaluate
        noise_types: Types of noise to evaluate
        occlusion_params: Occlusion parameters to evaluate
        results_dir: Directory to save results
        
    Returns:
        results: Dictionary containing evaluation results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate on clean data
    clean_acc, clean_preds, targets = evaluate_model(model, data_loader, device)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    
    results = {
        'clean_accuracy': clean_acc,
        'adversarial': {},
        'noise': {},
        'occlusion': {}
    }
    
    # Evaluate under adversarial attacks
    for attack_type in attack_types:
        results['adversarial'][attack_type] = {}
        
        for eps in epsilons:
            print(f"Evaluating {attack_type.upper()} attack with epsilon={eps}")
            acc, _, preds, _ = evaluate_under_attack(model, data_loader, device, attack_type, eps)
            results['adversarial'][attack_type][eps] = acc
            print(f"  Accuracy: {acc:.2f}%")
    
    # Evaluate under noise
    for noise_type in noise_types:
        results['noise'][noise_type] = {}
        
        for eps in epsilons:
            print(f"Evaluating {noise_type} noise with epsilon={eps}")
            acc, _, preds, _ = evaluate_under_noise(model, data_loader, device, noise_type, eps)
            results['noise'][noise_type][eps] = acc
            print(f"  Accuracy: {acc:.2f}%")
    
    # Evaluate under occlusion
    for params in occlusion_params:
        occlusion_type = params['type']
        occlusion_size = params['size']
        
        key = f"{occlusion_type}_{occlusion_size}"
        results['occlusion'][key] = {}
        
        print(f"Evaluating {occlusion_type} occlusion with size={occlusion_size}")
        acc, _, preds, _ = evaluate_under_occlusion(model, data_loader, device, occlusion_type, occlusion_size)
        results['occlusion'][key] = acc
        print(f"  Accuracy: {acc:.2f}%")
    
    # Plot accuracy vs epsilon for different attacks
    plot_robustness_curves(results, os.path.join(results_dir, 'robustness_curves.png'))
    
    return results


def plot_robustness_curves(results, save_path=None):
    """
    Plot accuracy vs epsilon for different perturbation types
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot for adversarial attacks
    for attack_type, attack_results in results['adversarial'].items():
        epsilons = sorted(list(attack_results.keys()))
        accuracies = [attack_results[eps] for eps in epsilons]
        plt.plot(epsilons, accuracies, 'o-', label=f'{attack_type.upper()} Attack')
    
    # Plot for noise
    for noise_type, noise_results in results['noise'].items():
        epsilons = sorted(list(noise_results.keys()))
        accuracies = [noise_results[eps] for eps in epsilons]
        plt.plot(epsilons, accuracies, 's-', label=f'{noise_type.capitalize()} Noise')
    
    # Add clean accuracy
    plt.axhline(y=results['clean_accuracy'], color='green', linestyle='--', label='Clean Accuracy')
    
    plt.title('Model Robustness to Different Perturbations')
    plt.xlabel('Epsilon (Perturbation Magnitude)')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 