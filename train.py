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
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    debug_mode = getattr(config, 'debug_mode', False)
    dataloaders, batch_transforms = get_dataloaders(config, device, debug_mode)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    
    # Подсчет параметров модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dataset sizes:")
    for partition, dataloader in dataloaders.items():
        print(f"    {partition}: {len(dataloader.dataset)} samples, batch_size={dataloader.batch_size}")
    print()
    
    logger.info(model)
    
    # Логируем параметры модели в CometML/WandB если доступно
    if writer is not None and hasattr(writer, 'exp') and writer.exp is not None:
        try:
            writer.exp.log_parameters({
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_name": config.model._target_.split('.')[-1],
            })
        except Exception as e:
            logger.warning(f"Не удалось залогировать параметры модели: {e}")
    else:
        logger.info("Writer не поддерживает логирование параметров или недоступен")

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()