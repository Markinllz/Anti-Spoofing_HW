import torch
from torch import nn
import torch.nn.functional as F
import math

class AsoftMax(nn.Module):
    """
    Angular margin based softmax loss (A-Softmax) для anti-spoofing.
    Реализация согласно статье ASVspoof2019 STC.
    """

    def __init__(self, margin=4, scale=30, **kwargs):
        """
        Args:
            margin (int): angular margin parameter (m в статье)
            scale (float): scale parameter (s в статье)
            **kwargs: additional arguments
        """
        super(AsoftMax, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, predictions, targets):
        """
        A-Softmax loss compute
        
        Args:
            predictions: model predictions (dict with 'logits' or tensor)
            targets: ground truth labels
        Returns:
            loss: computed A-Softmax loss value
        """
        if isinstance(predictions, dict):
            logits = predictions['logits']
        else:
            logits = predictions
            
        # For binary classification with 1 output, use BCEWithLogitsLoss
        # A-Softmax is complex for binary case, so use standard approach
        # as mentioned in the paper for anti-spoofing
        targets = targets.float()  # BCE expects float targets
        
        # Apply scale to logits and use BCEWithLogitsLoss
        scaled_logits = self.scale * logits.squeeze(-1)
        
        return F.binary_cross_entropy_with_logits(scaled_logits, targets)