"""
Тестовый скрипт для проверки исправлений в коде.
"""
import warnings
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

# Используем исправленный тренер
from src.trainer.trainer_fixed import Trainer

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Тест исправлений:
    1. LR scheduler вызывается в конце эпохи
    2. Gradient clipping отключен
    3. Добавлены проверки на NaN/Inf
    4. Улучшенное логирование
    """
    print("🔧 Тестируем исправления...")
    
    # Устанавливаем seed
    set_random_seed(config.trainer.seed)
    
    # Модифицируем конфигурацию для теста
    OmegaConf.set_struct(config, False)
    config.trainer.n_epochs = 3  # Только 3 эпохи для теста
    config.trainer.log_step = 10  # Чаще логируем
    config.optimizer.lr = 0.001  # Безопасный LR
    config.writer.run_name = "fixes-test"
    OmegaConf.set_struct(config, True)
    
    # Настройка логирования
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Устройство: {device}")
    
    # Создаем даталоадеры
    dataloaders, batch_transforms = get_dataloaders(config, device, debug_mode=False)
    
    # Создаем модель
    model = instantiate(config.model).to(device)
    print(f"🔢 Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # Создаем компоненты
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)
    
    # Создаем оптимизатор и планировщик
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    
    print(f"📉 Начальный LR: {lr_scheduler.get_last_lr()[0]}")
    print(f"📅 LR scheduler: уменьшение каждые {config.lr_scheduler.step_size} эпох на {config.lr_scheduler.gamma}")
    
    # Создаем ИСПРАВЛЕННЫЙ тренер
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
        skip_oom=True,
    )
    
    print("\n🚀 Запускаем тест с исправлениями...")
    print("🔧 Исправления:")
    print("   ✅ LR scheduler перенесен в конец эпохи")
    print("   ✅ Gradient clipping отключен")
    print("   ✅ Добавлены проверки на NaN/Inf")
    print("   ✅ Улучшенное логирование")
    print("-" * 50)
    
    # Тестируем один батч для проверки
    print("\n🧪 Тест одного батча:")
    train_dataloader = dataloaders["train"]
    batch = next(iter(train_dataloader))
    
    # Проверяем размеры данных
    print(f"📊 Размер батча: {batch['data_object'].shape}")
    print(f"📊 Метки: {batch['labels'].shape}, уникальные: {torch.unique(batch['labels'])}")
    
    # Тестируем forward pass
    model.eval()
    with torch.no_grad():
        batch = trainer.move_batch_to_device(batch)
        batch = trainer.transform_batch(batch)
        outputs = model(**batch)
        loss_dict = loss_function(**{**batch, **outputs})
        
        print(f"📊 Logits shape: {outputs['logits'].shape}")
        print(f"📊 Loss: {loss_dict['loss'].item():.6f}")
        print(f"📊 Logits range: [{outputs['logits'].min().item():.4f}, {outputs['logits'].max().item():.4f}]")
    
    print("\n✅ Тест одного батча прошел успешно!")
    
    # Запускаем полное обучение
    trainer.train()
    
    print("\n✅ Тест исправлений завершен!")

if __name__ == "__main__":
    main() 