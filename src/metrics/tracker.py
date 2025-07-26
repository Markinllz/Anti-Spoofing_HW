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
        
        # Добавляем хранилище для EER метрики
        self._eer_scores = []
        self._eer_labels = []

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        # Сбрасываем EER данные
        self._eer_scores = []
        self._eer_labels = []

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def update_eer(self, scores, labels):
        """
        Обновляет данные для вычисления EER метрики.
        
        Args:
            scores (torch.Tensor): предсказанные скоры
            labels (torch.Tensor): истинные метки
        """
        self._eer_scores.extend(scores.detach().cpu().numpy())
        self._eer_labels.extend(labels.detach().cpu().numpy())

    def compute_eer(self):
        """
        Вычисляет EER на основе накопленных данных.
        """
        if not self._eer_scores:
            return 0.0
        
        from src.metrics.eer import EERMetric
        eer_metric = EERMetric()
        eer_value = eer_metric(scores=np.array(self._eer_scores), 
                              labels=np.array(self._eer_labels))
        return eer_value

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
        # Добавляем EER если есть данные
        if self._eer_scores:
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