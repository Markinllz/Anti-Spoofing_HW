from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        # Выполняем backward только в режиме обучения
        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Обновляем loss метрики
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # Обновляем EER метрику
        if "logits" in batch:
            scores = torch.softmax(batch["logits"], dim=1)[:, 1]
            labels = batch["labels"]
            metrics.update_eer(scores, labels)

        # Обновляем остальные метрики
        for met in metric_funcs:
            if met.name != "eer":
                try:
                    metric_value = met(**batch)
                    metrics.update(met.name, metric_value)
                except Exception as e:
                    continue
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        """
        # Логируем информацию о батче для writer
        if self.writer is not None:
            # Логируем learning rate
            if mode == "train" and self.lr_scheduler is not None:
                self.writer.add_scalar("learning_rate", self.lr_scheduler.get_last_lr()[0])
            
            # Логируем градиентную норму
            if mode == "train":
                grad_norm = self._get_grad_norm()
                self.writer.add_scalar("grad_norm", grad_norm)