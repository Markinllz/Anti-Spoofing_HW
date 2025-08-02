# Исправленная версия tracker.py
import pandas as pd
import numpy as np
from collections import defaultdict


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()
        
        
        self._eer_scores = []
        self._eer_labels = []

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        
        # Сброс EER данных
        self._eer_scores = []
        self._eer_labels = []
        print(f"🔄 MetricTracker сброшен. EER lists очищены.")

    def update(self, key, value, n=1):
        

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def update_eer(self, scores, labels):
        """
        Update EER scores and labels.
        
        Args:
            scores (torch.Tensor): prediction scores
            labels (torch.Tensor): ground truth labels
        """
        if scores.numel() == 0 or labels.numel() == 0:
            print(f"⚠️ update_eer: получены пустые тензоры")
            return
            
        # Конвертируем в numpy
        scores_np = scores.detach().cpu().numpy().flatten()
        labels_np = labels.detach().cpu().numpy().flatten()
        
        # Проверяем что размеры совпадают
        if len(scores_np) != len(labels_np):
            print(f"⚠️ update_eer: размеры не совпадают scores={len(scores_np)}, labels={len(labels_np)}")
            return
        
        # Проверяем валидность данных
        if np.any(np.isnan(scores_np)) or np.any(np.isinf(scores_np)):
            print(f"⚠️ update_eer: некорректные scores (nan/inf)")
            return
            
        # Добавляем данные
        self._eer_scores.extend(scores_np)
        self._eer_labels.extend(labels_np)

    def compute_eer(self):
        """
        Compute Equal Error Rate from accumulated scores and labels.
        """
        if not self._eer_scores or len(self._eer_scores) == 0:
            print(f"⚠️ compute_eer: нет данных для вычисления EER")
            return 0.0
        
        scores = np.array(self._eer_scores)
        labels = np.array(self._eer_labels)
        
        print(f"🧮 Вычисляем EER: {len(scores)} семплов")
        print(f"    Уникальные labels: {np.unique(labels)}")
        print(f"    Scores range: {scores.min():.4f} - {scores.max():.4f}")
        
        # Проверяем что есть оба класса
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"⚠️ compute_eer: только один класс {unique_labels}")
            return 0.0
        
        # Получаем уникальные пороги
        thresholds = np.unique(scores)
        if len(thresholds) < 2:
            print(f"⚠️ compute_eer: все scores одинаковые")
            return 0.5  # Random baseline
        
        # Вычисляем FAR и FRR для каждого порога
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            # Предсказания: 1 если score >= threshold, иначе 0
            predictions = (scores >= threshold).astype(int)
            
            # Вычисляем confusion matrix
            tp = np.sum((predictions == 1) & (labels == 1))
            tn = np.sum((predictions == 0) & (labels == 0))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            # Вычисляем FAR и FRR
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            far_values.append(far)
            frr_values.append(frr)
        
        # Находим точку, где FAR ≈ FRR
        far_values = np.array(far_values)
        frr_values = np.array(frr_values)
        
        # Находим индекс, где разность минимальна
        diff = np.abs(far_values - frr_values)
        min_idx = np.argmin(diff)
        
        # EER - это среднее FAR и FRR в этой точке
        eer = (far_values[min_idx] + frr_values[min_idx]) / 2
        
        print(f"✅ EER вычислен: {eer:.6f} (FAR={far_values[min_idx]:.6f}, FRR={frr_values[min_idx]:.6f})")
        
        return float(eer)

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        if key == "eer":
            return self.compute_eer()
        return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        result = dict(self._data.average)
        # Всегда включаем EER в результат
        result['eer'] = self.compute_eer()
        return result

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        keys = list(self._data.total.keys())
        if self._eer_scores:
            keys.append('eer')
        return keys