import torch
from torch import nn
import torch.nn.functional as F

class AsoftMax(nn.Module):
    """
    Сигмоида loss для anti-spoofing (как в статье).
    """

    def __init__(self, margin=4, scale=30, **kwargs):
        """
        Args:
            margin (float): margin parameter
            scale (float): scale parameter
            **kwargs: additional arguments (ignored for simplicity)
        """
        super(AsoftMax, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, batch, **kwargs):
        """
        Сигмоида loss compute (как в статье для 5% EER)
        
        Args:
            batch (dict): batch containing 'logits' and 'labels'
            **kwargs: дополнительные аргументы (игнорируются)
        Returns:
            losses (dict): dictionary loss
        """
        logits = batch['logits']
        labels = batch['labels']
        
        # Correct sigmoid loss as in paper
        # For binary classification use CrossEntropy with proper handling
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        return {"loss": F.cross_entropy(logits, labels)}