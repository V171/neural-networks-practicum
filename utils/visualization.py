"""
Функции для визуализации результатов обучения.
Использование: from utils.visualization import plot_training_curves, show_predictions
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, save_path=None):
    """
    Визуализация кривых обучения.
    
    Args:
        history: словарь с ключами 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: путь для сохранения графика (опционально)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Кривые потерь')
    axes[0].legend()
    axes[0].grid(True)
    
    # График точности
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Кривые точности')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    
    plt.show()


def show_predictions(model, dataloader, classes, device, num_images=10, save_path=None):
    """
    Визуализация предсказаний модели.
    
    Args:
        model: обученная модель
        dataloader: DataLoader с тестовыми данными
        classes: список названий классов
        device: устройство (cuda/cpu)
        num_images: количество изображений для показа
        save_path: путь для сохранения (опционально)
    """
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Определяем количество строк и столбцов
    rows = (num_images + 4) // 5
    cols = min(num_images, 5)
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = [axes]
    
    for i in range(num_images):
        ax = axes[i // cols] if rows > 1 else axes
        ax = ax[i % cols] if rows > 1 else ax[i]
        
        img = images[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze()
            cmap = 'gray'
        else:  # RGB
            img = np.transpose(img, (1, 2, 0))
            img = (img * 0.5 + 0.5).clip(0, 1)  # Денормализация
            cmap = None
        
        ax.imshow(img, cmap=cmap)
        
        true_label = classes[labels[i].item()]
        pred_label = classes[predicted[i].item()]
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        ax.set_title(f'Истина: {true_label}\nПредск: {pred_label}', 
                    color=color, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Изображение сохранено: {save_path}")
    
    plt.show()


def plot_feature_maps(feature_maps, layer_name='', save_path=None):
    """
    Визуализация карт признаков свёрточного слоя.
    
    Args:
        feature_maps: тензор карт признаков (C, H, W)
        layer_name: название слоя для заголовка
        save_path: путь для сохранения (опционально)
    """
    num_maps = min(feature_maps.shape[0], 32)
    rows, cols = 4, 8
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.set_title(f'Фильтр {i}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Карты признаков: {layer_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Карты признаков сохранены: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, classes, save_path=None):
    """
    Визуализация матрицы ошибок.
    
    Args:
        cm: матрица ошибок (numpy array)
        classes: список названий классов
        save_path: путь для сохранения (опционально)
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Матрица ошибок сохранена: {save_path}")
    
    plt.show()
