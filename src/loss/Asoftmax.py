import torch
from torch import nn
import torch.nn.functional as F

class AsoftMax(nn.Module):
    """
    Простая сигмоида loss для anti-spoofing.
    """

    def __init__(self, margin=0.2, scale=15):
        super().__init__()
        # Параметры игнорируются для простоты

    def forward(self, batch, **kwargs):
        """
        Сигмоида loss compute
        
        Args:
            batch (dict): batch containing 'logits' and 'labels'
            **kwargs: дополнительные аргументы (игнорируются)
        Returns:
            losses (dict): dictionary loss
        """
        logits = batch['logits']
        labels = batch['labels']
        
        # Простой CrossEntropy loss (эквивалент сигмоиды для бинарной классификации)
        loss = F.cross_entropy(logits, labels)
        
        return {"loss": loss}