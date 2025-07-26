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
        
        self._eer_scores = []
        self._eer_labels = []

    def update(self, key, value, n=1):
        

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def update_eer(self, scores, labels):
     

        self._eer_scores.extend(scores.detach().cpu().numpy())
        self._eer_labels.extend(labels.detach().cpu().numpy())

    def compute_eer(self):

        if not self._eer_scores:
            return 0.0
        
        scores = np.array(self._eer_scores)
        labels = np.array(self._eer_labels)
        
        bona_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]
        
        if len(bona_scores) == 0 or len(spoof_scores) == 0:
            return 0.0
        
        # Вычисляем EER
        from src.metrics.eer import EERMetric
        eer_metric = EERMetric()
        eer, _ = eer_metric.compute_eer(bona_scores, spoof_scores)
        return eer

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