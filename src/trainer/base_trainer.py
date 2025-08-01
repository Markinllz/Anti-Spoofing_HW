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
            k: v for k, v in dataloaders.items() if k == "val"  # ТОЛЬКО val, НЕ test!
        }
        
        # test dataloader отдельно - только для финального inference
        self.test_dataloader = dataloaders.get("test", None)

       
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

      

        self.save_period = (
            self.cfg_trainer.save_period
        )  
        self.val_period = self.cfg_trainer.get("val_period", 1)
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  
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

     
        self.writer = writer

       
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )


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
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # Основной лог с тренировочными метриками
            log = {"epoch": epoch}
            log.update(result)

            # ВАЛИДАЦИЯ В КОНЦЕ ЭПОХИ (согласно val_period)
            val_log = {}
            if "val" in self.evaluation_dataloaders and (epoch % self.val_period == 0 or epoch == self.epochs):
                val_results = self._run_validation(epoch, is_mid_epoch=False)
                
                # НЕ добавляем префикс val_ здесь, он уже есть в конфиге monitor
                val_log = {f"val_{k}": v for k, v in val_results.items()}
                
                # Добавляем валидационные метрики в log с префиксом для логгера
                log.update(val_log)

            # evaluate model performance according to configured metric
            # ИСПОЛЬЗУЕМ ВАЛИДАЦИОННЫЕ МЕТРИКИ для model selection (если есть валидация в эту эпоху)!
            monitor_logs = val_log if val_log else result
            best, stop_process, not_improved_count = self._monitor_performance(monitor_logs, not_improved_count=not_improved_count)
            early_stop_count = not_improved_count

            best_ckpt_path = str(self.checkpoint_dir / "model_best.pth")
            
            # Логируем все метрики в основной лог  
            for key, value in log.items():
                self.logger.info(f"    {key:15s}: {value}")

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False, only_best=False)

            if early_stop_count == 0:
                print(f"Сохранение лучшей модели: {best_ckpt_path}")
                self._save_checkpoint(epoch, save_best=True, only_best=True)

            if early_stop_count >= self.config.trainer.get("early_stop", float("inf")):
                print(f"Ранняя остановка на эпохе {epoch}")
                break
                
        # ФИНАЛЬНАЯ ВАЛИДАЦИЯ
        print(f"\n🎯 Финальная валидация:")
        if "val" in self.evaluation_dataloaders:
            final_results = self._run_validation(self.epochs, is_mid_epoch=False, is_final=True)
        
        # TEST ЗАПУСКАЕТСЯ ОТДЕЛЬНО - НЕ ЗДЕСЬ!
        if self.test_dataloader is not None:
            print(f"\n⚠️ TEST набор доступен для финального inference")
            print(f"   Используйте inference.py для запуска теста")
        
        print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  
                    continue
                else:
                    raise e



         
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                print(f"  🏋️ Эпоха {epoch} {self._progress(batch_idx)} - текущий batch_loss: {batch['loss'].item():.6f}")
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                
                # Выводим текущие метрики в консоль каждые 50 батчей
                print(f"  📊 Текущие метрики (шаг {batch_idx}):")
                if "loss" in last_train_metrics:
                    print(f"    train_loss: {last_train_metrics['loss']:.6f}")
                if "eer" in last_train_metrics:
                    print(f"    train_eer: {last_train_metrics['eer']:.6f}")
                for metric_name, metric_value in last_train_metrics.items():
                    if metric_name not in ["loss", "eer"]:
                        print(f"    train_{metric_name}: {metric_value:.6f}")
                
                self.train_metrics.reset()
                
            # ВАЛИДАЦИЯ В СЕРЕДИНЕ ЭПОХИ (на половине эпохи)
            mid_epoch_step = self.epoch_len // 2
            if (batch_idx > 0 and 
                batch_idx == mid_epoch_step and 
                "val" in self.evaluation_dataloaders):
                print(f"\n🔄 Переключение на валидацию в середине эпохи...")
                self._run_validation(epoch, step=batch_idx, is_mid_epoch=True)
                print(f"🔄 Возврат к обучению...")
                # Возвращаем модель в режим обучения
                self.model.train()
                self.is_train = True
                
            if batch_idx + 1 >= self.epoch_len:
                break

        train_results = last_train_metrics
        print(f"\n🏋️ Тренировочные метрики эпохи {epoch}:")
        # Выводим loss первым, потом остальные метрики
        if "loss" in train_results:
            print(f"    train_loss: {train_results['loss']:.6f}")
        for metric_name, metric_value in train_results.items():
            if metric_name != "loss":  # loss уже вывели
                print(f"    train_{metric_name}: {metric_value:.6f}")
        print(f"   ✅ Эпоха обучения завершена")
        
        if self.writer is not None:
            self.writer.set_step(epoch, "train")
            for metric_name, metric_value in train_results.items():
                self.writer.add_scalar(f"train_{metric_name}_epoch", metric_value)

        # ⬅️ Возвращаем словарь с метриками эпохи
        return train_results

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            # Логирование по шагам валидации убрано - происходит в _run_validation
            # с правильными префиксами и step значениями
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _run_validation(self, epoch, step=None, is_mid_epoch=False, is_final=False):
        """
        Запуск валидации с корректным логированием.
        
        Args:
            epoch (int): номер эпохи
            step (int, optional): номер шага (для валидации в середине эпохи)
            is_mid_epoch (bool): True если валидация в середине эпохи
            is_final (bool): True если финальная валидация
        Returns:
            val_results (dict): результаты валидации
        """
        if "val" not in self.evaluation_dataloaders:
            return {}
            
        # Определяем тип валидации для логирования
        if is_final:
            validation_type = "финальная"
        elif is_mid_epoch:
            validation_type = "в середине эпохи"
        else:
            validation_type = "в конце эпохи"
            
        step_info = f" (шаг {step})" if step is not None else ""
        
        print(f"\n📊 Валидация {validation_type} {epoch}{step_info}:")
        
        val_dataloader = self.evaluation_dataloaders["val"]
        print(f"   📂 Количество валидационных батчей: {len(val_dataloader)}")
        
        val_results = self._evaluation_epoch(epoch, "val", val_dataloader)
        
        # Выводим метрики валидации в консоль
        if is_final:
            print(f"🎯 Финальные результаты валидации:")
        else:
            print(f"📊 Результаты валидации эпохи {epoch}{step_info}:")
        
        # Выводим loss первым, потом остальные метрики
        if "loss" in val_results:
            print(f"    val_loss: {val_results['loss']:.6f}")
        for metric_name, metric_value in val_results.items():
            if metric_name != "loss":  # loss уже вывели
                print(f"    val_{metric_name}: {metric_value:.6f}")
        
        print(f"   ✅ Валидация завершена")
        
        # Логируем в writer с правильными префиксами
        if self.writer is not None:
            if is_final:
                # Финальная валидация
                self.writer.set_step(epoch, "val")
                for metric_name, metric_value in val_results.items():
                    self.writer.add_scalar(f"val_{metric_name}_final", metric_value)
            elif is_mid_epoch and step is not None:
                # Валидация в середине эпохи - логируем по шагу
                self.writer.set_step((epoch - 1) * self.epoch_len + step, "val")
                for metric_name, metric_value in val_results.items():
                    self.writer.add_scalar(f"val_{metric_name}_mid_epoch", metric_value)
            else:
                # Валидация в конце эпохи - логируем по эпохе
                self.writer.set_step(epoch, "val")
                for metric_name, metric_value in val_results.items():
                    self.writer.add_scalar(f"val_{metric_name}_epoch", metric_value)
        
        return val_results

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        if self.batch_transforms is not None:
            transform_type = "train" if self.is_train else "inference"
            transforms = self.batch_transforms.get(transform_type)
            if transforms is not None:
                for transform_name in transforms.keys():
                    batch[transform_name] = transforms[transform_name](
                        batch[transform_name]
                    )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
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
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)