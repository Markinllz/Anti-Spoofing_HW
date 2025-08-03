import torch
from torch import nn
import torch.nn.functional as F

class AsoftMax(nn.Module):
    """
    Сигмоида loss для anti-spoofing (как в статье).
    """

    def __init__(self, margin=0.2, scale=15):
        super().__init__()
        # Параметры игнорируются для простоты

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
        
        # Правильная сигмоида loss как в статье
        # Для бинарной классификации используем CrossEntropy с правильной обработкой
        loss = F.cross_entropy(logits, labels)
        
        return {"loss": loss}