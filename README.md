# Neural Networks Workshop

Практикум по нейронным сетям (50 академических часов)

**Авторы:** доц. А.А. Часовских, с.н.с. В.С. Половников

## Описание курса

Данный курс предназначен для студентов старших курсов и аспирантов направления "Прикладная математика и информатика". Курс охватывает ключевые темы современного глубокого обучения: от базовых свёрточных сетей до диффузионных моделей.

## Содержание

| № | Тема | Часы |
|---|------|------|
| 1 | Введение в CNN. Процесс обучения в PyTorch | 5 |
| 2 | Переобучение и регуляризация | 5 |
| 3 | Transfer Learning: Feature Extraction vs Fine-tuning | 5 |
| 4 | Оптимизация, Дистилляция и Mutual Learning | 5 |
| 5 | Детектирование объектов (YOLO) | 5 |
| 6 | Трансформеры в компьютерном зрении (ViT) | 5 |
| 7 | Автоэнкодеры (AE) и латентные пространства | 5 |
| 8 | Вариационные автоэнкодеры (VAE) | 5 |
| 9 | Генеративно-состязательные сети (GAN) | 5 |
| 10 | Диффузионные модели (DDPM / LDM) | 5 |

**Итого:** 50 академических часов

## Структура репозитория

```
neural-networks-workshop/
├── README.md                           # Этот файл
├── notebooks/                          # Jupyter Notebooks для занятий
│   ├── 01_cnn_introduction.ipynb
│   ├── 02_overfitting_regularization.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_distillation_mutual_learning.ipynb
│   ├── 05_yolo_detection.ipynb
│   ├── 06_vision_transformer.ipynb
│   ├── 07_autoencoder.ipynb
│   ├── 08_vae.ipynb
│   ├── 09_gan.ipynb
│   └── 10_diffusion.ipynb
├── solutions/                          # Решения домашних заданий
├── scripts/                            # Вспомогательные скрипты
│   ├── utils.py                        # Утилиты для обучения
│   └── check_environment.py            # Проверка окружения
├── docs/                               # Документация
│   └── neural_networks_workshop.tex    # LaTeX версия практикума
└── data/                               # Примеры данных
```

## Быстрый старт

### Вариант 1: Google Colab (рекомендуется)

1. Откройте [Google Colab](https://colab.research.google.com/)
2. Выберите **GitHub** → вставьте URL репозитория
3. Откройте нужный ноутбук
4. Установите runtime: **Runtime → Change runtime type → GPU T4**

### Вариант 2: Локальный запуск

```bash
# Клонирование репозитория
git clone https://github.com/V171/neural-networks-workshop.git
cd neural-networks-workshop

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установка зависимостей
pip install torch torchvision matplotlib scikit-learn tqdm
pip install ultralytics  # для YOLO

# Запуск Jupyter
jupyter notebook notebooks/
```

### Проверка окружения

```bash
python scripts/check_environment.py
```

## Требования

- Python 3.9+
- PyTorch 2.0+
- Google аккаунт (для Colab)
- GPU (T4 или выше рекомендуется)

## Датасеты

Все датасеты загружаются автоматически:
- FashionMNIST (Занятия 1-2)
- CIFAR-10 (Занятия 4, 6)
- MNIST (Занятия 7, 10)
- Hymenoptera (Занятие 3, с fallback на CIFAR-10)

## Вклад в проект

Приветствуются:
- Сообщения об ошибках → [Issues](https://github.com/V171/neural-networks-workshop/issues)
- Предложения по улучшению → [Pull Requests](https://github.com/V171/neural-networks-workshop/pulls)
- Дополнительные примеры и решения

## Лицензия

Образовательные материалы. Использование разрешено в учебных целях.

## Контакты

Преподаватели:
- доц. А.А. Часовских
- с.н.с. В.С. Половников

---

**Spring 2026**
