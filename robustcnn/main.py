import os
import argparse
import torch
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import DatasetLoader
from models import get_model, get_trainer
from utils import evaluate_model, evaluate_under_attack, evaluate_under_noise, plot_confusion_matrix


def set_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train a robust image classifier')
    
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training and evaluation')
    
    
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['simple_cnn', 'lenet', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16'],
                       help='Model architecture to use')
    
    
    parser.add_argument('--training_mode', type=str, default='standard',
                       choices=['standard', 'adversarial', 'distillation', 'noise_augmentation'],
                       help='Training mode to use')
    parser.add_argument('--epochs', type=int, default=100,
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
                       choices=['fgsm', 'pgd', 'linf-pgd'],
                       help='Adversarial attack type for training')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Epsilon parameter for adversarial training')
    
    
    parser.add_argument('--noise_type', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'salt_and_pepper'],
                       help='Noise type for augmentation')
    
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU index to use')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    
    
    args = parse_args()
    
    
    set_seed(args.seed)
    
    
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.model}_{args.training_mode}")
    os.makedirs(save_dir, exist_ok=True)
    
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    print(f"Loading {args.dataset} dataset...")
    data_loader = DatasetLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    dataset_info = data_loader.get_dataset_info()
    
    
    print(f"Creating {args.model} model...")
    model = get_model(args.model, dataset_info)
    model = model.to(device)
    
    
    print(f"Training with {args.training_mode} mode...")
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
            'noise_type': args.noise_type,
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
    
    
    scheduler = ReduceLROnPlateau(trainer.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        scheduler=scheduler
    )
    
    
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    
    history_path = os.path.join(save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    
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
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    
    
    print("\nEvaluating on test set...")
    test_acc, test_preds, test_targets = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    
    class_names = [str(i) for i in range(dataset_info['num_classes'])]
    plot_confusion_matrix(
        test_targets, 
        test_preds, 
        class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    
    if args.training_mode in ['adversarial', 'distillation', 'noise_augmentation']:
        print("\nEvaluating robustness...")
        
        
        fgsm_acc, clean_acc, fgsm_preds, _ = evaluate_under_attack(
            model, test_loader, device, 'fgsm', args.epsilon
        )
        print(f"Accuracy under FGSM attack (ε={args.epsilon}): {fgsm_acc:.2f}%")
        
        
        pgd_acc, _, pgd_preds, _ = evaluate_under_attack(
            model, test_loader, device, 'pgd', args.epsilon
        )
        print(f"Accuracy under PGD attack (ε={args.epsilon}): {pgd_acc:.2f}%")
        
        
        noise_acc, _, noise_preds, _ = evaluate_under_noise(
            model, test_loader, device, 'gaussian', args.epsilon
        )
        print(f"Accuracy under Gaussian noise (ε={args.epsilon}): {noise_acc:.2f}%")
        
        
        robustness_results = {
            'clean_accuracy': clean_acc,
            'fgsm_accuracy': fgsm_acc,
            'pgd_accuracy': pgd_acc,
            'noise_accuracy': noise_acc,
            'epsilon': args.epsilon
        }
        
        with open(os.path.join(save_dir, 'robustness_results.json'), 'w') as f:
            json.dump(robustness_results, f)
    
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main() 