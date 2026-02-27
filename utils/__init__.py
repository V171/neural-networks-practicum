"""
Utils package for Neural Networks Practicum.

Доступные модули:
- training: функции для обучения и валидации
- visualization: визуализация результатов
- models: готовые архитектуры моделей
"""

from .training import train_epoch, validate_epoch, count_parameters, set_seed
from .visualization import plot_training_curves, show_predictions, plot_feature_maps
from .models import SimpleCNN, RegularizedCNN, get_resnet18, get_mobilenet_v2
from .models import save_model, load_model

__all__ = [
    'train_epoch', 'validate_epoch', 'count_parameters', 'set_seed',
    'plot_training_curves', 'show_predictions', 'plot_feature_maps',
    'SimpleCNN', 'RegularizedCNN', 'get_resnet18', 'get_mobilenet_v2',
    'save_model', 'load_model'
]
