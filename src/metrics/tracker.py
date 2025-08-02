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

    def update(self, key, value, n=1):
        

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def update_eer(self, scores, labels):
        """
        Update EER scores and labels.
        
        Args:
            scores (torch.Tensor): prediction scores (должны быть scores для bonafide класса)
            labels (torch.Tensor): ground truth labels (1 = bonafide, 0 = spoof)
        """
        if scores.numel() == 0 or labels.numel() == 0:
            return
            
        # Конвертируем в numpy
        scores_np = scores.detach().cpu().numpy().flatten()
        labels_np = labels.detach().cpu().numpy().flatten()
        
        # Проверяем что размеры совпадают
        if len(scores_np) != len(labels_np):
            return
        
        # Проверяем валидность данных
        if np.any(np.isnan(scores_np)) or np.any(np.isinf(scores_np)):
            return
            
        # Добавляем данные
        self._eer_scores.extend(scores_np)
        self._eer_labels.extend(labels_np)

    def compute_eer(self):
        """
        Compute Equal Error Rate from accumulated scores and labels using reference implementation.
        """
        if not self._eer_scores or len(self._eer_scores) == 0:
            return 0.0
        
        scores = np.array(self._eer_scores)
        labels = np.array(self._eer_labels)
        
        # Проверяем что есть оба класса
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        # Разделяем scores на bonafide и spoof
        bonafide_scores = scores[labels == 1]  # Настоящие образцы
        spoof_scores = scores[labels == 0]     # Поддельные образцы
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return 0.0
        
        # Используем эталонную реализацию
        eer, threshold = self._compute_eer_reference(bonafide_scores, spoof_scores)
        
        return float(eer)

    def _compute_eer_reference(self, bonafide_scores, spoof_scores):
        """
        Reference implementation of EER computation (эталонная реализация).
        
        Args:
            bonafide_scores: scores for bonafide (genuine) samples
            spoof_scores: scores for spoof (fake) samples
            
        Returns:
            tuple: (eer, threshold)
        """
        frr, far, thresholds = self._compute_det_curve(bonafide_scores, spoof_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        
        return float(eer), float(thresholds[min_index])

    def _compute_det_curve(self, target_scores, nontarget_scores):
        """
        Compute Detection Error Tradeoff (DET) curve (эталонная реализация).
        
        Args:
            target_scores: scores for target (bonafide) samples
            nontarget_scores: scores for nontarget (spoof) samples
            
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

        # False rejection rates (FRR): отклонение настоящих как поддельных
        frr = np.concatenate(
            (np.atleast_1d(0), tar_trial_sums / target_scores.size))
        
        # False acceptance rates (FAR): принятие поддельных как настоящих  
        far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                              nontarget_scores.size))
        
        # Thresholds are the sorted scores
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

        return frr, far, thresholds

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