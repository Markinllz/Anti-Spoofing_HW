from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch

class Trainer(BaseTrainer):
    """
    ИСПРАВЛЕННАЯ версия Trainer class.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Обработка батча с исправлениями.
        """
        # Перемещаем данные на устройство
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # Определяем метрики в зависимости от режима
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        # Forward pass через модель
        outputs = self.model(**batch)
        batch.update(outputs)

        # Вычисляем loss
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        # ОБУЧЕНИЕ ТОЛЬКО В ТРЕНИРОВОЧНОМ РЕЖИМЕ
        if self.is_train:
            # Проверяем, что loss корректный
            if torch.isnan(batch["loss"]) or torch.isinf(batch["loss"]):
                print(f"⚠️ WARNING: Некорректный loss обнаружен: {batch['loss'].item()}")
                return batch
            
            # Backward pass
            batch["loss"].backward()
            
            # ОТКЛЮЧЕН GRADIENT CLIPPING
            # self._clip_grad_norm()  # ← ОТКЛЮЧЕНО
            
            # Проверяем градиенты
            grad_norm = self._get_grad_norm()
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"⚠️ WARNING: Некорректные градиенты: {grad_norm}")
                return batch
            
            # Обновляем параметры
            self.optimizer.step()
            
            # LR SCHEDULER ВЫЗЫВАЕТСЯ В КОНЦЕ ЭПОХИ, НЕ ЗДЕСЬ!
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()  # ← УБРАНО! Это была главная ошибка!

        # Обновляем loss метрики
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                loss_value = batch[loss_name].item()
                if not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value))):
                    metrics.update(loss_name, loss_value)

        # Обновляем EER метрику
        if "logits" in batch:
            try:
                logits = batch["logits"]
                labels = batch["labels"]
                
                # Проверяем корректность logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"⚠️ WARNING: Некорректные logits")
                else:
                    scores = torch.softmax(logits, dim=1)[:, 1]
                    metrics.update_eer(scores, labels)
            except Exception as e:
                print(f"⚠️ WARNING: Ошибка при вычислении EER: {e}")

        # Обновляем остальные метрики
        for met in metric_funcs:
            if met.name != "eer":
                try:
                    metric_value = met(**batch)
                    if not (torch.isnan(torch.tensor(metric_value)) or torch.isinf(torch.tensor(metric_value))):
                        metrics.update(met.name, metric_value)
                except Exception as e:
                    continue

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Логирование батча с дополнительными проверками.
        """
        if self.writer is not None:
            # Логируем learning rate
            if mode == "train" and self.lr_scheduler is not None:
                try:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    self.writer.add_scalar("learning_rate", current_lr)
                except:
                    pass
            
            # Логируем градиентную норму
            if mode == "train":
                try:
                    grad_norm = self._get_grad_norm()
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        self.writer.add_scalar("grad_norm", grad_norm.item())
                except:
                    pass 