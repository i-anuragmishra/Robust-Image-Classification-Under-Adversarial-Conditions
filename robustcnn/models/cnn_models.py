import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    A simple CNN for image classification with basic architecture
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LeNet(nn.Module):
    """
    LeNet architecture for MNIST
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNetModel(nn.Module):
    """
    ResNet model with adjustable depth for image classification
    """
    def __init__(self, model_type='resnet18', in_channels=3, num_classes=10, pretrained=False):
        super(ResNetModel, self).__init__()
        
        # Dictionary to map model type to model constructor
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50
        }
        
        if model_type not in model_dict:
            raise ValueError(f"Model type {model_type} not supported. Choose from {list(model_dict.keys())}")
        
        # Initialize the model
        if pretrained:
            self.model = model_dict[model_type](weights='IMAGENET1K_V1')
        else:
            self.model = model_dict[model_type](weights=None)
        
        # Modify first conv layer if input channels don't match
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


class VGGModel(nn.Module):
    """
    VGG model with adjustable depth for image classification
    """
    def __init__(self, model_type='vgg11', in_channels=3, num_classes=10, pretrained=False):
        super(VGGModel, self).__init__()
        
        # Dictionary to map model type to model constructor
        model_dict = {
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19
        }
        
        if model_type not in model_dict:
            raise ValueError(f"Model type {model_type} not supported. Choose from {list(model_dict.keys())}")
        
        # Initialize the model
        if pretrained:
            self.model = model_dict[model_type](weights='IMAGENET1K_V1')
        else:
            self.model = model_dict[model_type](weights=None)
        
        # Modify first conv layer if input channels don't match
        if in_channels != 3:
            first_conv_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.model.features[0] = first_conv_layer
        
        # Replace final classifier layer
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


def get_model(model_name, dataset_info):
    """
    Factory function to get a model instance
    
    Args:
        model_name: Name of the model to instantiate
        dataset_info: Dictionary containing dataset information
            (input_channels, num_classes)
            
    Returns:
        model: Instantiated model
    """
    in_channels = dataset_info['input_channels']
    num_classes = dataset_info['num_classes']
    
    if model_name == 'simple_cnn':
        return SimpleCNN(in_channels, num_classes)
    elif model_name == 'lenet':
        return LeNet(in_channels, num_classes)
    elif model_name.startswith('resnet'):
        return ResNetModel(model_name, in_channels, num_classes)
    elif model_name.startswith('vgg'):
        return VGGModel(model_name, in_channels, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.") 