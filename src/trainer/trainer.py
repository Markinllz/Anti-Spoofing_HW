from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, metric_funcs):
        """
        Run batch through the model, compute metrics, and update the tracker.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): tracker that aggregates metrics
                over the dataset.
            metric_funcs (list): functions that computes metrics.

        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        
        # Подготовка для обучения
        if self.is_train:
            self.optimizer.zero_grad()
        
        batch.update(self.model(batch))
        batch.update(self.criterion(batch))

        # Обратное распространение и обновление весов
        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Update loss metrics
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # ИСПРАВЛЕННОЕ ВЫЧИСЛЕНИЕ EER
        if "logits" in batch and "labels" in batch:
            logits = batch["logits"]
            labels = batch["labels"]
            
            # Убеждаемся что размеры корректны
            if logits.dim() == 2 and logits.size(1) >= 2:
                # Получаем вероятности для класса bonafide (класс 0) - правильно для EER
                scores = torch.softmax(logits, dim=1)[:, 0]
                
                # Проверяем что есть данные
                if scores.numel() > 0 and labels.numel() > 0:
                    metrics.update_eer(scores, labels)
                else:
                    print(f"⚠️ Пустые данные для EER: scores.numel()={scores.numel()}, labels.numel()={labels.numel()}")
            else:
                print(f"⚠️ Неправильные размеры logits: {logits.shape}")
        else:
            print(f"⚠️ Отсутствуют необходимые ключи: logits={('logits' in batch)}, labels={('labels' in batch)}")

        # Update other metrics
        for met in metric_funcs:
            if met.name != "eer":
                try:
                    metrics.update(met.name, met(batch))
                except Exception as e:
                    print(f"⚠️ Ошибка в метрике {met.name}: {e}")

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.
        """
        pass