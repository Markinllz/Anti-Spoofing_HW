from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt:
            self._save_checkpoint(self._last_epoch, save_best=False, only_best=True)
            raise
        except Exception as e:
            raise

    def _train_process(self):
        """
        Full training logic.
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            self._train_epoch(epoch)

            # Валидируем только на val, test оставляем для инференса
            if "val" in self.evaluation_dataloaders:
                val_logs = {}
                dataloader = self.evaluation_dataloaders["val"]
                val_part_logs = self._evaluation_epoch(epoch, "val", dataloader)
                val_logs.update(val_part_logs)
                
                # Выводим финальные метрики валидации в консоль
                print(f"\n📊 Финальные метрики валидации эпохи {epoch}:")
                for metric_name, metric_value in val_logs.items():
                    print(f"    val_{metric_name}: {metric_value:.6f}")
                print()
                
                # Дополнительно логируем финальные метрики валидации в конце эпохи
                if self.writer is not None:
                    self.writer.set_step(epoch, "val")
                    for metric_name, metric_value in val_logs.items():
                        self.writer.add_scalar(f"val_{metric_name}_epoch", metric_value)

                # log best so far
                if self.mnt_mode != "off":
                    improved = self._monitor_performance(val_logs, not_improved_count)
                    if improved:
                        not_improved_count = 0
                    else:
                        not_improved_count += 1

                if self.mnt_mode != "off" and not_improved_count > self.early_stop:
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # Получаем количество батчей для определения середины эпохи
        total_batches = len(self.train_dataloader)
        mid_epoch_batch = total_batches // 2
        
        pbar = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = self.process_batch(batch, self.train_metrics)
                self._log_batch(batch_idx, batch, "train")
                
                # Выводим лосс в консоль каждые log_step батчей
                if batch_idx % self.log_step == 0:
                    loss_key = self.config.writer.loss_names[0]
                    current_loss = self.train_metrics.avg(loss_key)
                    current_eer = self.train_metrics.avg("eer") if self.train_metrics._eer_scores else 0.0
                    
                    print(f"[Epoch {epoch}, Batch {batch_idx}] Loss: {current_loss:.6f}, EER: {current_eer:.6f}")
                    
                    # Логируем метрики в CometML каждые log_step батчей
                    if self.writer is not None:
                        # Вычисляем общий шаг как epoch * num_batches + batch_idx
                        global_step = (epoch - 1) * len(self.train_dataloader) + batch_idx
                        self.writer.set_step(global_step, "train")
                        self.writer.add_scalar("train_loss_batch", current_loss)
                        if self.train_metrics._eer_scores:
                            self.writer.add_scalar("train_eer_batch", current_eer)
                
                # Валидация в середине эпохи (если val_period = 1)
                if batch_idx == mid_epoch_batch and "val" in self.evaluation_dataloaders:
                    # Сохраняем текущий режим
                    was_training = self.is_train
                    was_model_training = self.model.training
                    
                    # В debug режиме пропускаем валидацию в середине эпохи для упрощения
                    debug_mode = getattr(self.config, 'debug_mode', False)
                    if debug_mode:
                        print("🔧 Debug mode: пропускаем валидацию в середине эпохи")
                        continue
                    
                    # Создаем временный MetricTracker для валидации в середине эпохи
                    mid_epoch_metrics = MetricTracker(
                        *self.config.writer.loss_names,
                        *[m.name for m in self.metrics["inference"]],
                        writer=self.writer,
                    )
                    
                    val_logs = {}
                    dataloader = self.evaluation_dataloaders["val"]
                    val_part_logs = self._evaluation_epoch(epoch, "val", dataloader)
                    val_logs.update(val_part_logs)
                    
                    # Логируем метрики валидации в середине эпохи
                    self._log_scalars(self.evaluation_metrics)
                    
                    # Дополнительно логируем валидационные метрики в середине эпохи
                    if self.writer is not None:
                        self.writer.set_step(batch_idx, "val")
                        for metric_name, metric_value in val_logs.items():
                            self.writer.add_scalar(f"val_{metric_name}_mid_epoch", metric_value)
                    
                    # Выводим метрики валидации в консоль
                    print(f"    Валидация в середине эпохи {epoch}:")
                    for metric_name, metric_value in val_logs.items():
                        print(f"        {metric_name}: {metric_value:.6f}")
                    print()
                    
                    # Возвращаем режим обучения
                    self.is_train = was_training
                    self.model.train(was_model_training)
                    
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                    
        self._log_scalars(self.train_metrics)
        
        # Дополнительно логируем финальные метрики тренировки
        train_results = self.train_metrics.result()
        print(f"\n📊 Финальные метрики тренировки эпохи {epoch}:")
        for metric_name, metric_value in train_results.items():
            print(f"    train_{metric_name}: {metric_value:.6f}")
        
        if self.writer is not None:
            self.writer.set_step(epoch, "train")
            for metric_name, metric_value in train_results.items():
                self.writer.add_scalar(f"train_{metric_name}_epoch", metric_value)

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch.

        Args:
            epoch (int): Current epoch number.
            part (str): Name of the data part.
            dataloader (DataLoader): Dataloader for validation.

        Returns:
            dict: Dictionary with validation logs.
        """
        # Проверяем, включен ли debug_mode (One Batch Test)
        debug_mode = getattr(self.config, 'debug_mode', False)
        
        if debug_mode:
            # В режиме отладки валидация тоже должна "обучаться" для переобучения
            print(f"🔧 Debug mode: валидация в режиме обучения для переобучения")
            self.is_train = True  # Оставляем в режиме обучения
            self.model.train()    # Модель в режиме обучения
            use_gradients = True  # Разрешаем градиенты
        else:
            # Обычный режим валидации
            self.is_train = False
            self.model.eval()
            use_gradients = False
        
        self.evaluation_metrics.reset()
        
        context_manager = torch.no_grad() if not use_gradients else torch.enable_grad()
        
        with context_manager:
            pbar = tqdm(dataloader, desc=f"Validation {part} Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    batch = self.process_batch(batch, self.evaluation_metrics)
                    # Убираем _log_batch чтобы не было лишних сообщений
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        # Возвращаем режим обучения (если не debug)
        if not debug_mode:
            self.is_train = True
        
        # Логируем метрики валидации (основной способ)
        self._log_scalars(self.evaluation_metrics)
                
        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Monitor the performance and save the best model.

        Args:
            logs (dict): Dictionary with validation logs.
            not_improved_count (int): Number of epochs without improvement.

        Returns:
            bool: True if the model improved.
        """
        if self.mnt_mode == "off":
            return False

        try:
            current = logs[self.mnt_metric]
        except KeyError:
            return False

        if self.mnt_mode == "min":
            improved = current < self.mnt_best
        else:
            improved = current > self.mnt_best

        if improved:
            self.mnt_best = current
            self._save_checkpoint(self._last_epoch, save_best=True)

        return improved

    def move_batch_to_device(self, batch):
        """
        Move batch to device.

        Args:
            batch (dict): Batch to move to device.

        Returns:
            dict: Batch on device.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transform batch using batch transforms.

        Args:
            batch (dict): Batch to transform.

        Returns:
            dict: Transformed batch.
        """
        if self.batch_transforms is not None:
            for transform_name, transform in self.batch_transforms.items():
                if transform_name in batch:
                    batch[transform_name] = transform(batch[transform_name])
        return batch

    def _clip_grad_norm(self):
        """
        Clip gradient norm.
        """
        if self.cfg_trainer.get("max_grad_norm") is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg_trainer.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Get gradient norm.

        Args:
            norm_type (int): Type of norm.

        Returns:
            float: Gradient norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm

    def _progress(self, batch_idx):
        """
        Print progress.

        Args:
            batch_idx (int): Current batch index.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): Current batch index.
            batch (dict): Batch data.
            mode (str): Mode (train or validation).
        """
        pass

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Log scalars to the experiment tracker.

        Args:
            metric_tracker (MetricTracker): Metric tracker.
        """
        if self.writer is None:
            return
            
        for metric_name, metric_value in metric_tracker.result().items():
            # Добавляем префикс в зависимости от типа метрик
            if metric_tracker == self.train_metrics:
                log_name = f"train_{metric_name}"
            elif metric_tracker == self.evaluation_metrics:
                log_name = f"val_{metric_name}"
            else:
                log_name = metric_name
                
            self.writer.add_scalar(log_name, metric_value)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save checkpoint.

        Args:
            epoch (int): Current epoch number.
            save_best (bool): Whether to save the best model.
            only_best (bool): Whether to save only the best model.
        """
        arch = type(self.model).__name__

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # ← ЭТО НЕПРАВИЛЬНО!

        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (self.checkpoint_dir).exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            del state["optimizer"], state["lr_scheduler"], state["config"]
            torch.save(state, best_path + ".tmp")
            import os
            os.replace(best_path + ".tmp", best_path)
        elif not only_best:
            torch.save(state, filename)
            del state["optimizer"], state["lr_scheduler"], state["config"]
            torch.save(state, filename + ".tmp")
            import os
            os.replace(filename + ".tmp", filename)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoint.

        Args:
            resume_path (str): Path to checkpoint.
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of checkpoint. "
                "This may create an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"] != self.config["optimizer"]:
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _from_pretrained(self, pretrained_path):
        """
        Load pretrained model.

        Args:
            pretrained_path (str): Path to pretrained model.
        """
        pretrained_path = str(pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])