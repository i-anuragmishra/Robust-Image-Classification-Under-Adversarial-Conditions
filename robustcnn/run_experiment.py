import os
import argparse
import torch
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from robustcnn.data import DatasetLoader
from robustcnn.models import get_model, get_trainer
from robustcnn.utils import (
    evaluate_model, evaluate_under_attack, evaluate_under_noise, evaluate_under_occlusion,
    plot_confusion_matrix, plot_roc_curve, visualize_adversarial_examples,
    evaluate_model_robustness, plot_robustness_curves
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run robust image classification experiments')
    
    
    parser.add_argument('--mode', type=str, default='train_and_evaluate', 
                        choices=['train_and_evaluate', 'evaluate_only'],
                        help='Experiment mode')
    
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training and evaluation')
    
    
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['simple_cnn', 'lenet', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16'],
                       help='Model architecture to use')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to load a pre-trained model (for evaluate_only mode)')
    
    
    parser.add_argument('--training_mode', type=str, default='standard',
                       choices=['standard', 'adversarial', 'distillation', 'noise_augmentation'],
                       help='Training mode to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    
    
    parser.add_argument('--attack', type=str, default='fgsm',
                       choices=['fgsm', 'pgd', 'linf-pgd', 'cw'],
                       help='Adversarial attack type')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Epsilon parameter for perturbations')
    
    
    parser.add_argument('--attacks', type=str, nargs='+', default=['fgsm', 'pgd'],
                       help='List of attacks to evaluate')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.03, 0.05, 0.1],
                       help='List of epsilons to evaluate')
    parser.add_argument('--noise_types', type=str, nargs='+', default=['gaussian', 'uniform', 'salt_and_pepper'],
                       help='List of noise types to evaluate')
    
    
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize')
    
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU index to use')
    parser.add_argument('--result_dir', type=str, default='./experiment_results',
                       help='Directory to save experiment results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name (default: auto-generated based on parameters)')
    
    return parser.parse_args()


def set_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(args, device, data_loader, dataset_info, experiment_dir):
    
    print("=" * 80)
    print(f"TRAINING MODEL: {args.model} with {args.training_mode} training mode")
    print("=" * 80)
    
    
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    
    model = get_model(args.model, dataset_info)
    model = model.to(device)
    
    
    trainer_kwargs = {
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer_type': args.optimizer
    }
    
    if args.training_mode == 'adversarial':
        trainer_kwargs.update({
            'attack_type': args.attack,
            'epsilon': args.epsilon
        })
    elif args.training_mode == 'noise_augmentation':
        trainer_kwargs.update({
            'noise_type': args.noise_types[0],  
            'epsilon': args.epsilon
        })
    
    trainer = get_trainer(
        training_mode=args.training_mode,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **trainer_kwargs
    )
    
    
    print(f"\nTraining for {args.epochs} epochs (early stopping patience: {args.patience})...")
    start_time = time.time()
    
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    
    model_path = os.path.join(experiment_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    
    with open(os.path.join(experiment_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    
    plot_training_curves(history, os.path.join(experiment_dir, 'training_curves.png'))
    
    return model, test_loader


def load_model(args, device, dataset_info):
    
    print("=" * 80)
    print(f"LOADING PRE-TRAINED MODEL FROM: {args.model_path}")
    print("=" * 80)
    
    model = get_model(args.model, dataset_info)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_trained_model(args, model, data_loader, device, experiment_dir):
    
    print("=" * 80)
    print("EVALUATING MODEL ROBUSTNESS")
    print("=" * 80)
    
    
    if isinstance(data_loader, torch.utils.data.DataLoader):
        test_loader = data_loader
    else:
        test_loader = data_loader.get_single_loader(train=False)
    
    dataset_info = data_loader.get_dataset_info() if hasattr(data_loader, 'get_dataset_info') else None
    
    
    eval_dir = os.path.join(experiment_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    
    print("\nStandard evaluation on clean data...")
    clean_acc, clean_preds, clean_targets = evaluate_model(model, test_loader, device)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    
    
    if dataset_info:
        class_names = [str(i) for i in range(dataset_info['num_classes'])]
        plot_confusion_matrix(
            clean_targets, 
            clean_preds, 
            class_names,
            save_path=os.path.join(eval_dir, 'confusion_matrix.png')
        )
        
        
        plot_roc_curve(
            model,
            test_loader,
            device,
            dataset_info['num_classes'],
            save_path=os.path.join(eval_dir, 'roc_curve.png')
        )
    
    
    attack_results = {}
    
    
    for attack_type in args.attacks:
        print(f"\nEvaluating against {attack_type.upper()} attack with epsilon={args.epsilon}...")
        attack_dir = os.path.join(eval_dir, f'attack_{attack_type}')
        os.makedirs(attack_dir, exist_ok=True)
        
        attack_acc, _, attack_preds, attack_targets = evaluate_under_attack(
            model, test_loader, device, attack_type, args.epsilon
        )
        
        print(f"Accuracy under {attack_type.upper()} attack: {attack_acc:.2f}%")
        attack_results[attack_type] = attack_acc
        
        
        if dataset_info:
            plot_confusion_matrix(
                attack_targets, 
                attack_preds, 
                class_names,
                save_path=os.path.join(attack_dir, 'confusion_matrix.png')
            )
        
        
        small_loader = get_small_loader(data_loader, args.batch_size)
        visualize_adversarial_examples(
            model,
            small_loader,
            device,
            attack_type=attack_type,
            epsilon=args.epsilon,
            num_examples=args.num_examples,
            save_path=os.path.join(attack_dir, 'adversarial_examples.png')
        )
    
    
    noise_results = {}
    
    
    for noise_type in args.noise_types:
        print(f"\nEvaluating against {noise_type} noise with epsilon={args.epsilon}...")
        noise_dir = os.path.join(eval_dir, f'noise_{noise_type}')
        os.makedirs(noise_dir, exist_ok=True)
        
        noise_acc, _, noise_preds, noise_targets = evaluate_under_noise(
            model, test_loader, device, noise_type, args.epsilon
        )
        
        print(f"Accuracy under {noise_type} noise: {noise_acc:.2f}%")
        noise_results[noise_type] = noise_acc
        
        
        if dataset_info:
            plot_confusion_matrix(
                noise_targets, 
                noise_preds, 
                class_names,
                save_path=os.path.join(noise_dir, 'confusion_matrix.png')
            )
    
    
    occlusion_results = {}
    
    
    occlusion_types = ['block', 'random_blocks', 'gaussian_patch']
    occlusion_sizes = [4, 8]
    
    for occlusion_type in occlusion_types:
        for occlusion_size in occlusion_sizes:
            print(f"\nEvaluating against {occlusion_type} occlusion with size={occlusion_size}...")
            occlusion_dir = os.path.join(eval_dir, f'occlusion_{occlusion_type}_{occlusion_size}')
            os.makedirs(occlusion_dir, exist_ok=True)
            
            occl_acc, _, occl_preds, occl_targets = evaluate_under_occlusion(
                model, test_loader, device, occlusion_type, occlusion_size
            )
            
            print(f"Accuracy under {occlusion_type} occlusion (size={occlusion_size}): {occl_acc:.2f}%")
            occlusion_results[f"{occlusion_type}_{occlusion_size}"] = occl_acc
            
            
            if dataset_info:
                plot_confusion_matrix(
                    occl_targets, 
                    occl_preds, 
                    class_names,
                    save_path=os.path.join(occlusion_dir, 'confusion_matrix.png')
                )
    
    
    print("\nPerforming comprehensive robustness evaluation across epsilon values...")
    robustness_dir = os.path.join(eval_dir, 'robustness')
    os.makedirs(robustness_dir, exist_ok=True)
    
    
    occlusion_params = [
        {'type': 'block', 'size': 4},
        {'type': 'block', 'size': 8},
        {'type': 'random_blocks', 'size': 4}
    ]
    
    
    small_test_loader = get_small_evaluation_loader(test_loader, subset_size=1000)
    
    
    robustness_results = evaluate_model_robustness(
        model,
        small_test_loader,
        device,
        attack_types=args.attacks,
        epsilons=args.epsilons,
        noise_types=args.noise_types,
        occlusion_params=occlusion_params,
        results_dir=robustness_dir
    )
    
    
    all_results = {
        'clean_accuracy': clean_acc,
        'attack_results': attack_results,
        'noise_results': noise_results,
        'occlusion_results': occlusion_results,
        'robustness_results': robustness_results,
        'model': args.model,
        'dataset': args.dataset,
        'training_mode': args.training_mode if hasattr(args, 'training_mode') else 'unknown',
        'attack_epsilon': args.epsilon,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    
    results_path = os.path.join(experiment_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nEvaluation completed and results saved!")
    print(f"Results saved to: {results_path}")
    
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    
    print("\nAdversarial attack results:")
    for attack_type, acc in attack_results.items():
        print(f"  {attack_type.upper()} attack (ε={args.epsilon}): {acc:.2f}%")
    
    print("\nNoise robustness results:")
    for noise_type, acc in noise_results.items():
        print(f"  {noise_type} noise (ε={args.epsilon}): {acc:.2f}%")
    
    print("\nOcclusion robustness results:")
    for occlusion_key, acc in occlusion_results.items():
        occlusion_type, size = occlusion_key.split('_')
        print(f"  {occlusion_type} occlusion (size={size}): {acc:.2f}%")
    
    return all_results


def get_small_loader(data_loader, batch_size):
    
    if isinstance(data_loader, torch.utils.data.DataLoader):
        return data_loader
    
    vis_batch_size = min(batch_size, 20)  
    return data_loader.get_single_loader(train=False, batch_size=vis_batch_size)


def get_small_evaluation_loader(data_loader, subset_size=1000):
    
    if len(data_loader.dataset) <= subset_size:
        return data_loader
    
    
    subset_indices = torch.randperm(len(data_loader.dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(data_loader.dataset, subset_indices)
    
    
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        num_workers=data_loader.num_workers if hasattr(data_loader, 'num_workers') else 2,
        pin_memory=True
    )
    
    return subset_loader


def plot_training_curves(history, save_path):
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    
    args = parse_args()
    
    
    set_seed(args.seed)
    
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.mode == 'train_and_evaluate':
            args.experiment_name = f"{args.dataset}_{args.model}_{args.training_mode}_{timestamp}"
        else:
            args.experiment_name = f"{args.dataset}_{args.model}_eval_{timestamp}"
    
    
    experiment_dir = os.path.join(args.result_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    
    print(f"Loading {args.dataset} dataset...")
    data_loader = DatasetLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    dataset_info = data_loader.get_dataset_info()
    
    
    if args.mode == 'train_and_evaluate':
        
        model, test_loader = train_model(args, device, data_loader, dataset_info, experiment_dir)
    else:
        
        if args.model_path is None:
            raise ValueError("Model path must be provided for evaluate_only mode")
        
        model = load_model(args, device, dataset_info)
        test_loader = data_loader.get_single_loader(train=False)
    
    
    evaluate_trained_model(args, model, test_loader, device, experiment_dir)
    
    print(f"\nExperiment completed! Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main() 