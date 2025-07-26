import numpy as np
import torch
from abc import abstractmethod

class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        raise NotImplementedError()

class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER) metric.
    Ожидает в batch два поля:
        - 'logits': torch tensor с предсказанными logits
        - 'labels': torch tensor с метками (1 — bona fide, 0 — spoof)
    """

    def __init__(self, name="eer"):
        super().__init__(name=name)

    def __call__(self, **batch):
        # Получаем logits и метки
        logits = batch["logits"]
        labels = batch["labels"]

        # Приводим к numpy, если это torch.Tensor
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()

        # Получаем скоры из logits (softmax)
        import torch.nn.functional as F
        if hasattr(logits, "detach"):
            # Если это tensor, используем softmax
            scores = F.softmax(logits, dim=-1)[:, 1]  # Берем вероятность bona fide
        else:
            # Если это numpy, конвертируем обратно в tensor
            logits_tensor = torch.from_numpy(logits)
            scores_tensor = F.softmax(logits_tensor, dim=-1)
            scores = scores_tensor[:, 1].numpy()  # Берем вероятность bona fide

        # bona fide = 1, spoof = 0
        bona_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        if len(bona_scores) == 0 or len(spoof_scores) == 0:
            return 0.0

        eer, _ = self.compute_eer(bona_scores, spoof_scores)
        return eer

    @staticmethod
    def compute_det_curve(target_scores, nontarget_scores):
        n_scores = target_scores.size + nontarget_scores.size
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

        indices = np.argsort(all_scores, kind='mergesort')
        labels = labels[indices]

        tar_trial_sums = np.cumsum(labels)
        nontarget_trial_sums = nontarget_scores.size - \
            (np.arange(1, n_scores + 1) - tar_trial_sums)

        frr = np.concatenate(
            (np.atleast_1d(0), tar_trial_sums / target_scores.size))
        far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                              nontarget_scores.size))
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
        return frr, far, thresholds

    @classmethod
    def compute_eer(cls, bona_scores, spoof_scores):
        frr, far, thresholds = cls.compute_det_curve(bona_scores, spoof_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return eer, thresholds[min_index]