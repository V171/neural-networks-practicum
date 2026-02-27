"""
Готовые архитектуры моделей для практикума.
Использование: from utils.models import SimpleCNN, get_resnet18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """
    Простая свёрточная нейронная сеть для классификации.
    
    Архитектура:
    - 3 свёрточных блока (Conv -> ReLU -> MaxPool)
    - 2 полносвязных слоя с Dropout
    
    Args:
        in_channels: количество входных каналов (1 для grayscale, 3 для RGB)
        num_classes: количество классов классификации
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class RegularizedCNN(nn.Module):
    """
    Свёрточная сеть с регуляризацией (BatchNorm + Dropout).
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(RegularizedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 128 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_resnet18(num_classes=10, pretrained=True):
    """
    Создаёт ResNet18, адаптированную для CIFAR-10 (32x32).
    
    Args:
        num_classes: количество классов
        pretrained: использовать предобученные веса
    
    Returns:
        model: модель ResNet18
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Адаптация под маленькие изображения (32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    
    return model


def get_mobilenet_v2(num_classes=10, pretrained=True):
    """
    Создаёт MobileNetV2, адаптированную для CIFAR-10.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    return model


def save_model(model, path, optimizer=None, epoch=None, best_acc=None):
    """
    Сохранение модели с метаданными.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if best_acc is not None:
        checkpoint['best_acc'] = best_acc
    
    torch.save(checkpoint, path)
    print(f"Модель сохранена: {path}")


def load_model(model, path, optimizer=None):
    """
    Загрузка модели.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Модель загружена: {path} (epoch={epoch}, best_acc={best_acc:.2f}%)")
    return model, optimizer, epoch, best_acc
