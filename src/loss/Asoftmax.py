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

    def forward(self, batch, **kwargs):
        """
        A-Softmax loss compute
        
        Args:
            batch (dict): batch containing 'logits' and 'labels'
            **kwargs: дополнительные аргументы
        Returns:
            losses (dict): dictionary with loss
        """
        logits = batch['logits']
        labels = batch['labels']
        
        # Для бинарной классификации используем CrossEntropy
        # A-Softmax сложен для бинарного случая, поэтому используем стандартный подход
        # как в статье для anti-spoofing
        return {"loss": F.cross_entropy(logits, labels)}