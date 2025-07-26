import torch
from torch import nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Простой CrossEntropy loss для стабильного обучения
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        CrossEntropy loss compute
        
        Args:
            logits (Tensor): model output predictions (batch_size, num_classes)
            labels (Tensor): ground truth labels (batch_size,)
            **kwargs: дополнительные аргументы (игнорируются)
        Returns:
            losses (dict): dictionary loss
        """
        
        # Используем обычный CrossEntropy loss для стабильности
        loss = F.cross_entropy(logits, labels)
        
        return {"loss": loss}
