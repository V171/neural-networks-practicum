"""
Вспомогательные функции для обучения нейронных сетей.
Использование: from utils.training import train_epoch, validate_epoch
"""

import torch
from tqdm import tqdm


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Обучение модели на одной эпохе.
    
    Args:
        model: нейронная сеть (nn.Module)
        loader: DataLoader с обучающими данными
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство (cuda/cpu)
    
    Returns:
        epoch_loss: средняя потеря за эпоху
        epoch_acc: точность в процентах
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    """
    Валидация модели на тестовых данных.
    
    Args:
        model: нейронная сеть (nn.Module)
        loader: DataLoader с тестовыми данными
        criterion: функция потерь
        device: устройство (cuda/cpu)
    
    Returns:
        epoch_loss: средняя потеря за эпоху
        epoch_acc: точность в процентах
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def count_parameters(model):
    """Подсчёт обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """Фиксация random seed для воспроизводимости."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
