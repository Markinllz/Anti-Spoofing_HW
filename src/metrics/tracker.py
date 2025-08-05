# Исправленная версия tracker.py
import pandas as pd
import numpy as np
from collections import defaultdict


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None, metrics=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
            metrics (list): list of metric objects for special handling
        """
        self.writer = writer
        self.metrics = metrics or []
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        
        # Reset metric objects that have reset method
        for metric in self.metrics:
            if hasattr(metric, 'reset'):
                metric.reset()

    def update(self, key, value, n=1):
        """
        Update metric with new value.
        
        Args:
            key (str): metric name
            value (float): metric value  
            n (int): number of samples
        """
        # For EER, we don't want to average - we want the final computed value
        if key == "eer":
            self._data.loc[key, "average"] = value
        else:
            self._data.loc[key, "total"] += value * n
            self._data.loc[key, "counts"] += n
            self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data.average[key]

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return dict(self._data.average)

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return list(self._data.total.keys())


class EERMetricTracker:
    """
    Специальный трекер для EER который накапливает все предсказания за эпоху
    и вычисляет EER только в конце эпохи.
    """

    def __init__(self, writer=None):
        """
        Args:
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
        """
        self.writer = writer
        self.all_scores = []
        self.all_labels = []
        self.reset()

    def reset(self):
        """
        Reset EER tracker.
        """
        self.all_scores = []
        self.all_labels = []

    def update(self, scores, labels):
        """
        Accumulate scores and labels for EER computation.
        
        Args:
            scores (torch.Tensor): prediction scores
            labels (torch.Tensor): ground truth labels
        """
        self.all_scores.extend(scores.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())

    def compute_eer(self):
        """
        Compute EER from accumulated scores and labels.
        
        Returns:
            float: EER value
        """
        if len(self.all_scores) == 0:
            return 0.0
        
        scores = np.array(self.all_scores)
        labels = np.array(self.all_labels)
        
        # Separate bonafide and spoof scores
        bonafide_scores = scores[labels == 0]
        spoof_scores = scores[labels == 1]
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return 0.0
        
        # Compute DET curve
        frr, far, thresholds = self._compute_det_curve(bonafide_scores, spoof_scores)
        
        # Find EER point
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        
        return float(eer)

    def _compute_det_curve(self, target_scores, nontarget_scores):
        """
        Compute DET curve (exactly as in EERMetric).
        
        Args:
            target_scores: scores for bonafide samples  
            nontarget_scores: scores for spoof samples
            
        Returns:
            tuple: (frr, far, thresholds)
        """
        n_scores = target_scores.size + nontarget_scores.size
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

        # Sort labels based on scores
        indices = np.argsort(all_scores, kind='mergesort')
        labels = labels[indices]

        # Compute false rejection and false acceptance rates
        tar_trial_sums = np.cumsum(labels)
        nontarget_trial_sums = nontarget_scores.size - \
            (np.arange(1, n_scores + 1) - tar_trial_sums)

        # false rejection rates
        frr = np.concatenate(
            (np.atleast_1d(0), tar_trial_sums / target_scores.size))
        far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                              nontarget_scores.size))  # false acceptance rates
        # Thresholds are the sorted scores
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

        return frr, far, thresholds