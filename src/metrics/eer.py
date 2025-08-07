import torch
import numpy as np
from typing import Dict, Any

from src.metrics.base_metric import BaseMetric


class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER) metric for audio anti-spoofing.
    Uses exact reference implementation from ASVspoof.
    Accumulates all predictions over epoch and computes EER correctly.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: additional arguments
        """
        super(EERMetric, self).__init__()
        self.name = "eer"
        self.reset()

    def reset(self):
        """
        Reset accumulated scores and labels.
        """
        self.all_scores = []
        self.all_labels = []

    def __call__(self, batch) -> float:
        """
        Accumulate scores and labels for EER computation.
        
        Args:
            batch: input batch containing scores and labels
            
        Returns:
            float: EER value (computed from all accumulated data)
        """
        if 'scores' in batch:
            scores = batch['scores']
        elif 'logits' in batch:
            logits = batch['logits']
            # For P2SGradLoss, logits are cosine angles in [-1, 1] range
            # We use the bonafide cosine angle as the score
            # Higher cosine angle = higher probability of being bonafide
            scores = logits[:, 0]  # Use first column as bonafide score
            # Convert from [-1, 1] to [0, 1] range for EER
            scores = (scores + 1) / 2
        else:
            return 0.0
        
        labels = batch['labels']
        
        # Check that we have data
        if scores.numel() == 0 or labels.numel() == 0:
            return 0.0
        
        # Accumulate scores and labels
        self.all_scores.extend(scores.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        
        # Compute EER from all accumulated data
        return self.compute_eer_from_accumulated()

    def compute_eer_from_accumulated(self):
        """
        Compute EER from all accumulated scores and labels.
        
        Returns:
            float: EER value
        """
        if len(self.all_scores) == 0:
            return 0.0
        
        scores = np.array(self.all_scores)
        labels = np.array(self.all_labels)
        
        # Separate bonafide and spoof scores
        # labels: 1 = bonafide, 0 = spoof
        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            return 0.0
        
        # Compute DET curve
        frr, far, thresholds = self.compute_det_curve(bonafide_scores, spoof_scores)
        
        # Find EER point
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        
        return float(eer)

    def compute_eer(self, scores: torch.Tensor, labels: torch.Tensor):
        """
        Returns equal error rate (EER) and the corresponding threshold.
        
        Args:
            scores (torch.Tensor): prediction scores (cosine angles for P2SGradLoss)
            labels (torch.Tensor): ground truth labels (0 = spoof, 1 = bonafide)
            
        Returns:
            tuple: (eer, threshold)
        """
       
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
       
        bonafide_scores = scores_np[labels_np == 1]  
        other_scores = scores_np[labels_np == 0]
        
     
        if len(bonafide_scores) == 0 or len(other_scores) == 0:
            return 0.0, 0.0
        
       
        frr, far, thresholds = self.compute_det_curve(bonafide_scores, other_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        
        return float(eer), float(thresholds[min_index])

    def compute_det_curve(self, target_scores, nontarget_scores):
        """
        Compute DET curve for EER calculation.
        
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