import torch
import numpy as np
from typing import Dict, Any

from src.metrics.base_metric import BaseMetric


class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER) metric for audio anti-spoofing.
    Использует точно эталонную реализацию из ASVspoof.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: additional arguments
        """
        super(EERMetric, self).__init__()
        self.name = "eer"

    def __call__(self, batch) -> float:
        """
        Compute EER metric.
        
        Args:
            batch: input batch containing scores and labels
            
        Returns:
            float: EER value
        """
        if 'scores' in batch:
            scores = batch['scores']
        elif 'logits' in batch:
            logits = batch['logits']
            scores = torch.softmax(logits, dim=1)[:, 0]
        else:
            return 0.0
        
        labels = batch['labels']
        
        
        eer, _ = self.compute_eer(scores, labels)
        
        return eer

    def compute_eer(self, scores: torch.Tensor, labels: torch.Tensor):
        """
        Returns equal error rate (EER) and the corresponding threshold.
        
        Args:
            scores (torch.Tensor): prediction scores (probability of bonafide)
            labels (torch.Tensor): ground truth labels (0 = bonafide, 1 = spoof)
            
        Returns:
            tuple: (eer, threshold)
        """
       
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
       
        bonafide_scores = scores_np[labels_np == 0]  
        other_scores = scores_np[labels_np == 1]
        
     
        if len(bonafide_scores) == 0 or len(other_scores) == 0:
            return 0.0, 0.0
        
       
        frr, far, thresholds = self.compute_det_curve(bonafide_scores, other_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        
        return float(eer), float(thresholds[min_index])

    def compute_det_curve(self, target_scores, nontarget_scores):
        """
        Точная копия эталонной реализации compute_det_curve.
        
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