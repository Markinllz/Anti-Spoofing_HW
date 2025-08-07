import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    """
    Binary Cross Entropy Loss with Sigmoid activation.
    Suitable for binary classification tasks.
    """
    
    def __init__(self, **kwargs):
        super(SigmoidLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: model predictions (logits before sigmoid)
            targets: ground truth labels (0 or 1)
        Returns:
            loss: computed BCE loss value
        """
        if isinstance(predictions, dict):
            logits = predictions['logits']
        else:
            logits = predictions
            
        # Ensure targets are float for BCE
        targets = targets.float()
        
        # For binary classification, squeeze to match target size
        if logits.dim() > 1:
            logits = logits.squeeze(-1)  # Remove last dimension to match target
        
        return self.bce_loss(logits, targets) 