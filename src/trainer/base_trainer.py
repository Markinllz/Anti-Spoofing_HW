from abc import abstractmethod

import torch
import numpy as np
from numpy import inf
from torch.nn.utils import clip_grad_norm_

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
            k: v for k, v in dataloaders.items() if k in ["eval"]
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
            metrics=self.metrics["train"],
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
            metrics=self.metrics["inference"],
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

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)
            
            # Additional save to my_model.pth every 5 epochs
            if epoch % 5 == 0:
                self._save_my_model(epoch)

            if stop_process:  # early_stop
                break

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
        
        print(f"\nEpoch {epoch}/{self.epochs} | Batches: {self.epoch_len}")
        
        # Accumulate metrics for log_step batches
        step_losses = []
        all_train_predictions = []
        all_train_labels = []
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Set batch_idx for debug purposes
            self.batch_idx = batch_idx
            
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # Accumulate metrics for current batch
            if "loss" in batch:
                step_losses.append(batch["loss"].item())
            
            # Store predictions for EER calculation
            all_train_predictions.append(batch["predictions"])
            all_train_labels.append(batch["labels"])
            
            # Update progress bar each batch
            progress = int(((batch_idx + 1) / self.epoch_len) * 100)
            filled = int(progress / 5)  # 20 blocks for 100%
            bar = "=" * filled + "-" * (20 - filled)
            print(f"\rEpoch {epoch} [{bar}] {progress}% ({batch_idx + 1}/{self.epoch_len})", end="")

            # Log and output statistics every log_step batches (NO EER)
            if (batch_idx + 1) % self.log_step == 0:
                # Calculate averages for last log_step batches
                avg_loss = sum(step_losses[-self.log_step:]) / len(step_losses[-self.log_step:]) if step_losses else 0
                
                # Output statistics (loss only) - output to console as before
                print(f"\nStatistics for batches {max(0, batch_idx + 1 - self.log_step)}-{batch_idx + 1}:")
                print(f"    Average Loss: {avg_loss:.6f}")
                
                # DO NOT log to writer here - only output to console
                # self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                # self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                # self._log_scalars(self.train_metrics)
                # self._log_batch(batch_idx, batch)
                
                # Reset metrics, but keep EER accumulation
                # Don't reset EER metric - let it accumulate over the entire epoch
                # We'll handle EER separately at the end of epoch
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break
        
        # Final progress bar at 100%
        print(f"\rEpoch {epoch} [====================] 100% ({self.epoch_len}/{self.epoch_len})")
        
        # Step the learning rate scheduler at the end of epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Final statistics for entire epoch (INCLUDING EER)
        if step_losses:
            epoch_avg_loss = sum(step_losses) / len(step_losses)
            print(f"Epoch {epoch} results:")
            print(f"    Average Loss for epoch: {epoch_avg_loss:.6f}")
            
            # Log loss per epochs in writer (always)
            self.writer.set_step(epoch, "train")
            self.writer.add_scalar("train/loss", epoch_avg_loss)
            self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
            
            # EER only at the end of epoch - compute from all predictions
            # Compute EER from all training predictions
            train_metrics = self._calculate_explicit_metrics(all_train_predictions, all_train_labels)
            train_eer = train_metrics['eer']
            print(f"    EER for epoch: {train_eer:.6f}")
            # Log EER in writer only at the end of epoch
            self.writer.add_scalar("train/eer", train_eer)
        
        print(f"Epoch {epoch} completed!")

        # Get final train metrics
        train_logs = self.train_metrics.result()
        logs = train_logs.copy()
        
        # Override EER with the correctly computed value
        if step_losses:
            logs['eer'] = train_eer  # Use the correctly computed EER
        
        # Log train metrics to CometML (loss already logged above)
        # self.writer.set_step(epoch, "train")  # Fixed: epoch instead of epoch * self.epoch_len
        # self._log_scalars(self.train_metrics)
        
        # Run eval evaluation every 2 epochs from the beginning
        for part, dataloader in self.evaluation_dataloaders.items():
            # Check if we should evaluate this partition
            should_evaluate = (epoch % 2 == 0)  # Eval every 2 epochs from the beginning
            
            if should_evaluate:
                val_logs = self._evaluation_epoch(epoch, part, dataloader)
                logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})
            else:
                # Skip evaluation for this partition
                print(f"Skipping {part} evaluation (not epoch {epoch})")

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on (dev or eval)
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        
        # Determine correct display name
        part_display = "testing"
        print(f"\n{part_display.capitalize()} on {part}...")
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            total_batches = len(dataloader)
            for batch_idx, batch in enumerate(dataloader):
                # Simple progress without excessive output
                progress = int(((batch_idx + 1) / total_batches) * 100)
                print(f"\r  {part_display.capitalize()}: {progress}% ({batch_idx + 1}/{total_batches})", end="")
                
                # Set batch_idx for debug purposes
                self.batch_idx = batch_idx
                
                batch = self.process_batch(
                    batch,
                    metrics=None,  # Don't use metrics tracker for evaluation
                    metric_funcs=None
                )
                
                # Store for explicit calculation
                all_predictions.append(batch["predictions"])
                all_labels.append(batch["labels"])
                all_losses.append(batch["loss"].item())
            
            # Final validation progress
            print(f"\r  {part_display.capitalize()}: 100% ({total_batches}/{total_batches})")
            
            # Calculate metrics explicitly
            metrics = self._calculate_explicit_metrics(all_predictions, all_labels)
            
            # Log evaluation metrics to CometML with correct step (always for eval)
            self.writer.set_step(epoch, part)  # Fixed: epoch instead of epoch * self.epoch_len
            self.writer.add_scalar(f"{part}/loss", metrics['loss'])
            self.writer.add_scalar(f"{part}/eer", metrics['eer'])

        # Print evaluation results nicely
        part_prefix = "Eval"
        print(f"Results of {part_display} {part}:")
        print(f"    {part_prefix} Loss: {metrics['loss']:.6f}")
        print(f"    {part_prefix} EER: {metrics['eer']:.6f}")

        return metrics

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
        device_tensors = self.cfg_trainer.device_tensors
        
        if not isinstance(batch, dict):
            return batch
            
        if isinstance(device_tensors, str):
            device_tensors = [device_tensors]
        elif not isinstance(device_tensors, list):
            device_tensors = list(device_tensors)
            
        for tensor_for_device in device_tensors:
            if tensor_for_device in batch:
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
        # Skip batch transforms if it's not a dict (e.g., CollateFn instance)
        if not isinstance(self.batch_transforms, dict):
            return batch
            
        # do batch transforms on device
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
            # For EER, use result() instead of avg() since EER is not averaged
            if metric_name == "eer":
                value = metric_tracker.result()[metric_name]
            else:
                value = metric_tracker.avg(metric_name)
            self.writer.add_scalar(f"{metric_name}", value)

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

    def _save_my_model(self, epoch):
        """
        Save model to my_model.pth in the root directory.
        
        Args:
            epoch (int): current epoch number.
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
        
        # Save to root directory (same level as src folder)
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = os.path.join(root_dir, "my_model.pth")
        
        torch.save(state, filename)
        self.logger.info(f"Saving my_model.pth at epoch {epoch}: {filename}")

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
        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _calculate_explicit_metrics(self, all_predictions, all_labels):
        """
        Calculate loss and EER explicitly without using trackers.
        
        Args:
            all_predictions: list of batch predictions from epoch
            all_labels: list of batch labels from epoch
            
        Returns:
            dict: {'loss': float, 'eer': float}
        """
        # Flatten predictions and labels from all batches
        all_scores = []
        all_labels_np = []
        total_loss = 0.0
        total_samples = 0
        
        for i, (batch_pred, batch_labels) in enumerate(zip(all_predictions, all_labels)):
            # Calculate loss for this batch
            if isinstance(batch_pred, dict):
                if 'logits' in batch_pred:
                    logits = batch_pred['logits']
                else:
                    continue  # Skip this batch
            else:
                # batch_pred is already logits tensor
                logits = batch_pred
            
            # Calculate loss for this batch
            batch_loss = self.criterion(logits, batch_labels)
            total_loss += batch_loss.item() * batch_labels.size(0)  # Weight by batch size
            total_samples += batch_labels.size(0)
            
            # Apply sigmoid for binary classification with 1 output
            # Ensure logits are 1D for binary classification
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            scores = torch.sigmoid(logits)  # Probability of bonafide class
            all_scores.extend(scores.detach().cpu().numpy())
            all_labels_np.extend(batch_labels.detach().cpu().numpy())
        
        if len(all_scores) == 0:
            print(f"  ERROR: No scores collected! This means predictions are empty.")
            return {'loss': 0.0, 'eer': 0.0}
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Convert to numpy arrays
        scores_np = np.array(all_scores)
        labels_np = np.array(all_labels_np)
        
        # Separate bonafide and spoof scores
        bonafide_scores = scores_np[labels_np == 1]
        spoof_scores = scores_np[labels_np == 0]
        
        # Check if scores are all the same (model not trained)
        if np.std(scores_np) < 1e-6:
            print(f"  WARNING: All scores are the same! Model may not be trained properly.")
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            print(f"  ERROR: No bonafide or spoof samples!")
            return {'loss': avg_loss, 'eer': 0.0}
        
        # Calculate EER using explicit function
        from src.metrics.eer import EERMetric
        eer_metric = EERMetric()
        eer, _ = eer_metric.compute_eer_from_arrays(bonafide_scores, spoof_scores)
        
        return {'loss': avg_loss, 'eer': float(eer)}