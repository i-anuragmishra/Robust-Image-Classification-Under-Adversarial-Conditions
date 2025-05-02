import os
import argparse
import torch
import json
from data import DatasetLoader
from models import get_model
from utils import (
    evaluate_model, evaluate_under_attack, evaluate_under_noise, evaluate_under_occlusion,
    plot_confusion_matrix, plot_roc_curve, visualize_adversarial_examples,
    evaluate_model_robustness
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate a robust image classifier')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['simple_cnn', 'lenet', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16'],
                       help='Model architecture to use')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    
    # Evaluation parameters
    parser.add_argument('--eval_mode', type=str, default='standard',
                       choices=['standard', 'robustness', 'visualization'],
                       help='Evaluation mode')
    
    # Attack parameters
    parser.add_argument('--attack', type=str, default='fgsm',
                       choices=['fgsm', 'pgd', 'linf-pgd', 'cw'],
                       help='Adversarial attack type for evaluation')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Epsilon parameter for evaluation')
    
    # Comprehensive evaluation parameters
    parser.add_argument('--attacks', type=str, nargs='+', default=['fgsm', 'pgd'],
                       help='List of attacks to evaluate')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.03, 0.05, 0.1],
                       help='List of epsilons to evaluate')
    parser.add_argument('--noise_types', type=str, nargs='+', default=['gaussian', 'uniform'],
                       help='List of noise types to evaluate')
    
    # Visualization parameters
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize')
    
    # Miscellaneous parameters
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU index to use')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f"eval_{args.dataset}_{args.model}_{args.eval_mode}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    data_loader = DatasetLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    test_loader = data_loader.get_single_loader(train=False)
    dataset_info = data_loader.get_dataset_info()
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(args.model, dataset_info)
    
    # Load trained model weights
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Standard evaluation
    print("\nPerforming standard evaluation...")
    test_acc, test_preds, test_targets = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Plot confusion matrix
    class_names = [str(i) for i in range(dataset_info['num_classes'])]
    plot_confusion_matrix(
        test_targets, 
        test_preds, 
        class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        model,
        test_loader,
        device,
        dataset_info['num_classes'],
        save_path=os.path.join(save_dir, 'roc_curve.png')
    )
    
    # Mode-specific evaluation
    if args.eval_mode == 'robustness':
        print("\nPerforming comprehensive robustness evaluation...")
        
        # Define occlusion parameters
        occlusion_params = [
            {'type': 'block', 'size': 4},
            {'type': 'block', 'size': 8},
            {'type': 'random_blocks', 'size': 4}
        ]
        
        # Evaluate model robustness
        results = evaluate_model_robustness(
            model,
            test_loader,
            device,
            attack_types=args.attacks,
            epsilons=args.epsilons,
            noise_types=args.noise_types,
            occlusion_params=occlusion_params,
            results_dir=save_dir
        )
        
        # Save comprehensive results
        with open(os.path.join(save_dir, 'comprehensive_results.json'), 'w') as f:
            json.dump(results, f)
            
    elif args.eval_mode == 'visualization':
        print("\nGenerating adversarial example visualizations...")
        
        # Create a smaller batch loader for visualization
        vis_batch_size = min(args.batch_size, 20)  # Smaller batch size for visualization
        vis_loader = DatasetLoader(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=vis_batch_size
        ).get_single_loader(train=False)
        
        # Visualize adversarial examples with different attacks
        for attack_type in args.attacks:
            print(f"Generating {attack_type.upper()} adversarial examples...")
            for eps in args.epsilons:
                visualize_adversarial_examples(
                    model,
                    vis_loader,
                    device,
                    attack_type=attack_type,
                    epsilon=eps,
                    num_examples=args.num_examples,
                    save_path=os.path.join(save_dir, f'{attack_type}_eps{eps}_examples.png')
                )
    
    else:  # Standard evaluation with specific attack
        print(f"\nEvaluating against {args.attack.upper()} attack with epsilon={args.epsilon}...")
        adv_acc, clean_acc, adv_preds, adv_targets = evaluate_under_attack(
            model, test_loader, device, args.attack, args.epsilon
        )
        print(f"Clean accuracy: {clean_acc:.2f}%")
        print(f"Accuracy under {args.attack.upper()} attack: {adv_acc:.2f}%")
        
        # Plot confusion matrix for adversarial examples
        plot_confusion_matrix(
            adv_targets, 
            adv_preds, 
            class_names,
            save_path=os.path.join(save_dir, f'{args.attack}_confusion_matrix.png')
        )
        
        # Evaluate under noise
        print(f"\nEvaluating against gaussian noise with epsilon={args.epsilon}...")
        noise_acc, _, noise_preds, noise_targets = evaluate_under_noise(
            model, test_loader, device, 'gaussian', args.epsilon
        )
        print(f"Accuracy under gaussian noise: {noise_acc:.2f}%")
        
        # Evaluate under occlusion
        print("\nEvaluating against block occlusion...")
        occl_acc, _, occl_preds, occl_targets = evaluate_under_occlusion(
            model, test_loader, device, 'block', occlusion_size=8
        )
        print(f"Accuracy under block occlusion: {occl_acc:.2f}%")
        
        # Save results
        results = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'noise_accuracy': noise_acc,
            'occlusion_accuracy': occl_acc,
            'attack_type': args.attack,
            'epsilon': args.epsilon
        }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main() 