import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with sigmoid activation.
    """
    
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: model predictions (logits)
            targets: ground truth labels (0 or 1)
        Returns:
            loss: computed loss value
        """
        if isinstance(predictions, dict):
            logits = predictions['logits']
        else:
            logits = predictions
            
        # Ensure targets are float
        targets = targets.float()
        
        return self.criterion(logits, targets)
