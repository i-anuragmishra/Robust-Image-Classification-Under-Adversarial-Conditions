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

## Key Results

We evaluated various robustness techniques across different datasets and model architectures. Here are our key findings:

### MNIST (LeNet)

Standard vs. Adversarial training under PGD attack:

| Epsilon | Standard Model | Adversarial Model | Improvement |
| ------- | -------------- | ----------------- | ----------- |
| 0.01    | 97.61%         | 98.06%            | +0.45%      |
| 0.1     | 96.94%         | 97.61%            | +0.67%      |
| 0.5     | 74.17%         | 84.75%            | +10.58%     |

### CIFAR-10 (SimpleCNN)

Standard vs. Adversarial training under PGD attack:

| Epsilon | Standard Model | Adversarial Model | Improvement |
| ------- | -------------- | ----------------- | ----------- |
| 0.01    | 29.46%         | 57.75%            | +28.29%     |
| 0.03    | 19.50%         | 48.80%            | +29.30%     |
| 0.05    | 12.01%         | 40.02%            | +28.01%     |

### CIFAR-10 (ResNet18)

Comparison of defense strategies under PGD attack (ε=0.03):

| Training Method        | Clean Accuracy | Under Attack | Improvement |
|------------------------|----------------|--------------|-------------|
| Standard               | 91.42%         | 23.18%       | -           |
| Adversarial (PGD)      | 87.65%         | 64.92%       | +41.74%     |
| Defensive Distillation | 89.78%         | 52.31%       | +29.13%     |
| Noise Augmentation     | 90.04%         | 48.76%       | +25.58%     |

### Key Insights

- **Dataset complexity matters**: MNIST models show natural robustness while CIFAR-10 models are highly vulnerable
- **Architecture impacts**: ResNet18 achieves higher clean accuracy but shows similar vulnerability patterns to SimpleCNN
- **Defense effectiveness**: PGD-based adversarial training provides the strongest protection (+41.74% improvement)
- **Clean accuracy trade-off**: Adversarial training slightly reduces clean accuracy but dramatically improves robustness
- **Class-specific vulnerabilities**: Natural objects (cats, dogs, birds) are more vulnerable than man-made objects

For full experimental details, see [README_EXPERIMENTS.md](README_EXPERIMENTS.md).

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

## Author and Course Information

- **Author**: Anurag Mishra
- **Email**: am2552@rit.edu
- **Institution**: Rochester Institute of Technology
- **Course**: IMGS 789 : Machine Learning for Difficult Data
- **Semester**: Spring 2025

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
