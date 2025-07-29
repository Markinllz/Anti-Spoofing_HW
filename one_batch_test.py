import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    One Batch Test script. Tests if the model can overfit on a small batch.
    This is useful for debugging the training pipeline.
    """
    print("🔍 Запуск One Batch Test...")
    print("📊 Ожидаем переобучение на маленьком датасете")
    
    # Устанавливаем seed для воспроизводимости
    set_random_seed(config.trainer.seed)
    
    # Добавляем debug_mode в конфигурацию
    OmegaConf.set_struct(config, False)
    config.debug_mode = True
    # Увеличиваем количество эпох для переобучения
    config.trainer.n_epochs = 100
    # Чаще логируем
    config.trainer.log_step = 1
    # Отключаем early stopping
    config.trainer.early_stop = 200
    OmegaConf.set_struct(config, True)
    
    # Настройка логирования
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    
    # Определяем устройство
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    
    print(f"🖥️  Используем устройство: {device}")
    
    # Создаем мини-даталоадеры (debug_mode=True)
    dataloaders, batch_transforms = get_dataloaders(config, device, debug_mode=True)
    
    # Проверяем, что все разделы используют одни и те же данные
    print("\n📊 Проверка данных в One Batch Test:")
    train_batch = next(iter(dataloaders["train"]))
    val_batch = next(iter(dataloaders["val"]))
    
    print(f"📝 Train batch shape: {train_batch['data_object'].shape}")
    print(f"📝 Val batch shape: {val_batch['data_object'].shape}")
    print(f"📝 Train labels: {train_batch['labels']}")
    print(f"📝 Val labels: {val_batch['labels']}")
    
    # Проверяем, идентичны ли данные
    data_identical = torch.equal(train_batch['data_object'], val_batch['data_object'])
    labels_identical = torch.equal(train_batch['labels'], val_batch['labels'])
    
    print(f"✅ Данные идентичны: {data_identical}")
    print(f"✅ Метки идентичны: {labels_identical}")
    
    if not (data_identical and labels_identical):
        print("⚠️  ВНИМАНИЕ: Данные train и val не идентичны!")
    else:
        print("🎯 Отлично: Train и Val используют одинаковые данные")
    
    # Создаем модель
    model = instantiate(config.model).to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Общее количество параметров: {total_params:,}")
    print(f"🎯 Обучаемых параметров: {trainable_params:,}")
    
    # Создаем функцию потерь и метрики
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)
    
    # Создаем оптимизатор и планировщик
    trainable_params_iter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params_iter)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    
    # Создаем тренер
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=None,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )
    
    print("🚀 Начинаем One Batch Test...")
    print("📈 Ожидаемое поведение:")
    print("   - Loss должен монотонно уменьшаться")
    print("   - Через 50-100 эпох loss должен быть < 0.01")
    print("   - Accuracy должна стремиться к 100%")
    print("   - EER должна стремиться к 0%")
    print("-" * 50)
    
    # Запускаем обучение
    trainer.train()
    
    print("✅ One Batch Test завершен!")

if __name__ == "__main__":
    main() 