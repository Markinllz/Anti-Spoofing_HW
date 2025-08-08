Anti-Spoofing System

Установка

Требования
- Python 3.9+
- PyTorch 2.2.0
- CUDA (опционально)

Установка зависимостей

1. Клонируйте репозиторий
```bash
git clone <repository-url>
cd Anti-Spoofing_Homework
```

2. Создайте виртуальное окружение
```bash
conda create -n anti-spoofing python=3.9
conda activate anti-spoofing
```

3. Установите зависимости
```bash
pip install -r requirements.txt
```

Использование

Обучение модели
```bash
python train.py
```

Инференс
```bash
python inference.py
```

Конфигурация

Проект использует Hydra. Основные файлы конфигурации находятся в `src/configs/`.

Структура проекта

```
Anti-Spoofing_Homework/
├── src/
│   ├── configs/
│   ├── datasets/
│   ├── logger/
│   ├── loss/
│   ├── metrics/
│   ├── model/
│   ├── trainer/
│   ├── transforms/
│   └── utils/
├── data/
├── checkpoints/
├── outputs/
└── wandb/
```
