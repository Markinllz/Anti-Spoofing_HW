import torch
import numpy as np
from typing import Dict, Any

from src.metrics.base_metric import BaseMetric


class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER) metric for audio anti-spoofing.
    Uses explicit calculation without tracking.
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
        Calculate EER directly from batch.
        
        Args:
            batch: input batch containing scores and labels
            
        Returns:
            float: EER value
        """
        if 'scores' in batch:
            scores = batch['scores']
        elif 'logits' in batch:
            logits = batch['logits']
            # For A-Softmax: apply softmax to get probabilities
            # For binary classification with 1 output, use sigmoid instead of softmax
            scores = torch.sigmoid(logits.squeeze(-1))  # Probability of bonafide class
        else:
            return 0.0
        
        labels = batch['labels']
        
        # Check that we have data
        if scores.numel() == 0 or labels.numel() == 0:
            return 0.0
        
        # Convert to numpy
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Separate bonafide and spoof scores
        # labels: 1 = bonafide, 0 = spoof
        bonafide_scores = scores_np[labels_np == 1]
        spoof_scores = scores_np[labels_np == 0]
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return 0.0
        
        # Calculate EER using explicit function
        eer, _ = self.compute_eer(bonafide_scores, spoof_scores)
        
        return float(eer)

    def compute_eer(self, bonafide_scores, other_scores):
        """
        Returns equal error rate (EER) and the corresponding threshold.
        Uses the same algorithm as calculate_eer.py
        """
        frr, far, thresholds = self.compute_det_curve(bonafide_scores, other_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return eer, thresholds[min_index]

    def compute_det_curve(self, target_scores, nontarget_scores):
        """
        Compute DET curve for EER calculation.
        Uses the same algorithm as calculate_eer.py
        
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

    def compute_eer_from_arrays(self, bonafide_scores, spoof_scores):
        """
        Compute EER from numpy arrays of bonafide and spoof scores.
        
        Args:
            bonafide_scores: numpy array of bonafide scores
            spoof_scores: numpy array of spoof scores
            
        Returns:
            tuple: (eer, threshold)
        """
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return 0.0, 0.0
        
        frr, far, thresholds = self.compute_det_curve(bonafide_scores, spoof_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        
        return float(eer), float(thresholds[min_index])