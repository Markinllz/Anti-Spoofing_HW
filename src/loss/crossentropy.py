import torch
import torch.nn as nn
from typing import Dict, Any


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss for audio anti-spoofing.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: additional arguments
        """
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        """
        Compute cross entropy loss.
        
        Args:
            **batch: input batch containing logits and labels
            
        Returns:
            Dict[str, torch.Tensor]: loss dictionary
        """
        # Получаем logits и labels
        logits = batch['logits']
        labels = batch['labels']
        
        # Проверяем размеры
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # Вычисляем потерю
        loss = self.criterion(logits, labels)
        
        return {
            'loss': loss
        }
