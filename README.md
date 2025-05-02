# RobustCNN: Adversarial Robustness Framework for Image Classification

RobustCNN is a comprehensive framework for evaluating the robustness of image classification models against various adversarial attacks and perturbations. This project helps researchers and practitioners analyze how well their models perform under adversarial conditions.

## Features

- **Multiple Datasets**: Support for MNIST, CIFAR-10, and extensible to other datasets
- **Various Model Architectures**: LeNet, SimpleCNN, ResNet (18/34/50), VGG (11/13/16)
- **Adversarial Attacks**:
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - Custom noise perturbations (Gaussian, Uniform, Salt & Pepper)
  - Occlusion attacks (Block, Random blocks)
- **Training Techniques**:
  - Standard training
  - Adversarial training
  - Defensive distillation
  - Noise augmentation
- **Comprehensive Evaluation**:
  - Clean accuracy
  - Robustness against attacks
  - Visualization of adversarial examples
  - Performance across varying perturbation strengths

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/robustcnn.git
   cd robustcnn
   ```

2. Install dependencies:

   ```bash
   pip install -r robustcnn/requirements.txt
   ```

3. (Optional) If you experience issues with advertorch, run:
   ```bash
   python fix_advertorch.py
   ```

## Quick Start

Run a quick test to verify the installation:

```bash
python quick_test.py
```

This will run a minimal experiment using MNIST with LeNet for 2 epochs to ensure everything is set up correctly.

## Basic Usage

Run a standard experiment:

```bash
python run_robust_cnn.py --dataset cifar10 --model resnet18 --training_mode standard --epochs 50
```

Train with adversarial examples:

```bash
python run_robust_cnn.py --dataset cifar10 --model resnet18 --training_mode adversarial --attack fgsm --epsilon 0.03 --epochs 50
```

Evaluate a pre-trained model:

```bash
python run_robust_cnn.py --mode evaluate_only --dataset cifar10 --model resnet18 --model_path experiment_results/your_model_dir/model.pth
```

## Command-line Arguments

### Basic Arguments:

- `--mode`: Experiment mode (`train_and_evaluate` or `evaluate_only`)
- `--dataset`: Dataset to use (`cifar10` or `mnist`)
- `--model`: Model architecture (`simple_cnn`, `lenet`, `resnet18`, etc.)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training and evaluation

### Training Arguments:

- `--training_mode`: Training approach (`standard`, `adversarial`, `distillation`, `noise_augmentation`)
- `--lr`: Learning rate
- `--optimizer`: Optimizer to use (`adam` or `sgd`)

### Adversarial Arguments:

- `--attack`: Attack type for adversarial training (`fgsm`, `pgd`, etc.)
- `--epsilon`: Perturbation strength
- `--attacks`: List of attacks for evaluation
- `--epsilons`: List of epsilon values for robustness evaluation

## Project Structure

```
robustcnn/
├── adversarial/        # Adversarial attack implementations
├── data/               # Dataset loading and processing
├── models/             # Model architectures and training
├── utils/              # Utility functions and evaluation metrics
├── experiments/        # Experiment configurations
├── results/            # Default directory for experiment results
├── run_experiment.py   # Main experiment runner
└── requirements.txt    # Project dependencies
```

## Examples

### Training a Robust ResNet18 on CIFAR-10

```bash
python run_robust_cnn.py \
  --dataset cifar10 \
  --model resnet18 \
  --training_mode adversarial \
  --attack pgd \
  --epsilon 0.03 \
  --epochs 100 \
  --batch_size 128 \
  --optimizer adam \
  --lr 0.001
```

### Comparing Different Defense Strategies

Train models with different defenses:

```bash
# Standard training
python run_robust_cnn.py --dataset cifar10 --model resnet18 --training_mode standard --experiment_name standard_resnet18

# Adversarial training (FGSM)
python run_robust_cnn.py --dataset cifar10 --model resnet18 --training_mode adversarial --attack fgsm --experiment_name fgsm_resnet18

# Adversarial training (PGD)
python run_robust_cnn.py --dataset cifar10 --model resnet18 --training_mode adversarial --attack pgd --experiment_name pgd_resnet18

# Noise augmentation
python run_robust_cnn.py --dataset cifar10 --model resnet18 --training_mode noise_augmentation --experiment_name noise_resnet18
```

## Visualization and Analysis

Experiment results are saved to the `experiment_results` directory (or the location specified by `--result_dir`), including:

- Model checkpoints
- Training history
- Evaluation metrics
- Visualizations of adversarial examples
- Confusion matrices
- Robustness curves

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
