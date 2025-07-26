# Anti-Spoofing System

Система для обнаружения спуфинга в аудио с использованием глубокого обучения.

## Установка

### Требования
- Python 3.9+
- PyTorch 2.2.0
- CUDA (опционально, для GPU)

### Установка зависимостей

1. **Клонируйте репозиторий:**
```bash
git clone <repository-url>
cd Anti-Spoofing_Homework
```

2. **Создайте виртуальное окружение:**
```bash
conda create -n anti-spoofing python=3.9
conda activate anti-spoofing
```

3. **Установите зависимости:**
```bash
# Основные зависимости
pip install -r requirements.txt

# Для разработки (опционально)
pip install -r requirements-dev.txt
```

## Использование

### Обучение модели

```bash
python train.py
```

### Инференс

```bash
# Установите API ключ CometML
export COMET_API_KEY="ваш_api_ключ"

# Запустите инференс
python inference.py

# Или используйте готовый скрипт
python test_inference.py
```

## Конфигурация

Проект использует Hydra для управления конфигурацией. Основные файлы конфигурации находятся в `src/configs/`.

### Настройка логирования

Для использования CometML:
```bash
export COMET_API_KEY="ваш_api_ключ"
python train.py
```

Для использования WandB:
```bash
wandb login
python train.py
```

## Структура проекта

```
Anti-Spoofing_Homework/
├── src/
│   ├── configs/          # Конфигурации Hydra
│   ├── datasets/         # Датсеты и загрузчики данных
│   ├── logger/           # Логгеры (CometML, WandB)
│   ├── loss/            # Функции потерь
│   ├── metrics/         # Метрики
│   ├── model/           # Модели
│   ├── trainer/         # Тренировка
│   ├── transforms/      # Трансформации данных
│   └── utils/           # Утилиты
├── data/                # Данные
├── checkpoints/         # Чекпоинты моделей
├── outputs/             # Выходы
└── wandb/              # Логи WandB
```
