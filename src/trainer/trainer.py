import torch
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, metric_funcs=None):
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
        batch = self.transform_batch(batch)

        if metric_funcs is None:
            metric_funcs = self.metrics["inference"]
            if self.is_train:
                metric_funcs = self.metrics["train"]
        
        if self.is_train:
            self.optimizer.zero_grad()

        # Extract data tensor from batch
        data_tensor = batch["data_object"]
        outputs = self.model(data_tensor)
        batch.update(outputs)

        loss = self.criterion(batch["logits"], batch["labels"])
        batch["loss"] = loss



        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            # LR scheduler step moved to end of epoch in base_trainer.py

        # Store predictions for explicit calculation
        # Store logits directly for EER calculation
        if "logits" in batch:
            batch["predictions"] = batch["logits"]
        else:
            batch["predictions"] = torch.zeros(batch["labels"].shape[0], 1)  # Fallback
        # Labels are already in batch

        # Only update metrics if metrics is not None
        if metrics is not None and hasattr(self.config, 'writer') and hasattr(self.config.writer, 'loss_names'):
            for loss_name in self.config.writer.loss_names:
                if loss_name in batch:
                    metrics.update(loss_name, batch[loss_name].item())

            for met in metric_funcs:
                metrics.update(met.name, met(batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        pass