# Neural Networks Practicum

Практикум по нейронным сетям для студентов направления «Прикладная математика и информатика».

## Аннотация

Данное учебно-методическое пособие предназначено для студентов старших курсов и аспирантов, изучающих курс «Нейронные сети». Практикум содержит 10 лабораторных работ, охватывающих ключевые темы современной глубокой обработки изображений: от базовых свёрточных сетей до современных генеративных моделей.

По завершении курса студенты будут способны:
- Разрабатывать и обучать нейронные сети для задач классификации и детекции
- Применять Transfer Learning для работы с малыми данными
- Реализовывать генеративные модели (VAE, GAN, Diffusion)
- Анализировать и интерпретировать результаты обучения
- Самостоятельно исследовать новые архитектуры и методы

## О курсе

- **Объём:** 50 академических часов
- **Формат:** 10 лабораторных работ
- **Инструменты:** Google Colab + PyTorch
- **Преподаватели:** д.ф.-м.н., профессор А.А. Часовских; к.ф.-м.н., с.н.с. В.С. Половников

## Структура репозитория

```
neural-networks-practicum/
├── notebooks/        # Шаблоны ноутбуков для занятий
├── solutions/        # Примеры решений (после дедлайнов)
├── utils/            # Вспомогательные функции
│   ├── training.py   # Функции обучения (train_epoch, validate_epoch)
│   ├── visualization.py  # Визуализация (графики, предсказания)
│   └── models.py     # Готовые архитектуры (SimpleCNN, ResNet18)
└── docs/             # Документация (PDF практикума)
```

## Темы занятий

| № | Тема | Часы |
|---|------|------|
| 1 | Введение в CNN. Процесс обучения в PyTorch | 5 |
| 2 | Переобучение и регуляризация | 5 |
| 3 | Transfer Learning (Feature Extraction, Fine-tuning) | 5 |
| 4 | Оптимизация, Дистилляция, Mutual Learning | 5 |
| 5 | Детектирование объектов (YOLO) | 5 |
| 6 | Vision Transformer (ViT) | 5 |
| 7 | Автоэнкодеры (AE) | 5 |
| 8 | Вариационные автоэнкодеры (VAE) | 5 |
| 9 | Генеративно-состязательные сети (GAN) | 5 |
| 10 | Диффузионные модели | 5 |

## Быстрый старт

### Клонирование репозитория

```bash
git clone https://github.com/V171/neural-networks-practicum.git
cd neural-networks-practicum
```

### Использование в Google Colab

```python
# Клонирование прямо в Colab
!git clone https://github.com/V171/neural-networks-practicum.git

# Импорт вспомогательных функций
import sys
sys.path.append('/content/neural-networks-practicum')
from utils import train_epoch, validate_epoch, plot_training_curves
```

### Локальная работа

```bash
# Установка зависимостей
pip install torch torchvision matplotlib tqdm

# Запуск Jupyter
jupyter notebook
```

## Использование utils

```python
from utils import train_epoch, validate_epoch, SimpleCNN, plot_training_curves

# Создание модели
model = SimpleCNN(in_channels=1, num_classes=10)

# Обучение
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

# Визуализация
plot_training_curves(history, save_path='training_curves.png')
```

## Требования

- Python 3.8+
- PyTorch 1.12+
- torchvision
- matplotlib
- tqdm
- numpy

## Полезные ссылки

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Google Colab](https://colab.research.google.com/)
- [Hugging Face](https://huggingface.co/)

## Обратная связь

Нашли ошибку или есть предложение? Создайте [Issue](https://github.com/V171/neural-networks-practicum/issues).

---

*Москва, 2026*
