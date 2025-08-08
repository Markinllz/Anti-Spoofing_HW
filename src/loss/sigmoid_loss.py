import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    """
    Cross Entropy Loss for binary classification with 2 outputs.
    Suitable for binary classification tasks with 2 logits (spoof, bonafide).
    """
    
    def __init__(self, **kwargs):
        super(SigmoidLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: model predictions (logits [batch_size, 2])
            targets: ground truth labels (0 for spoof, 1 for bonafide)
        Returns:
            loss: computed CrossEntropy loss value
        """
        if isinstance(predictions, dict):
            logits = predictions['logits']
        else:
            logits = predictions
            
        # Ensure targets are long for CrossEntropyLoss
        targets = targets.long()
        
        # CrossEntropyLoss expects [batch_size, num_classes] and [batch_size]
        return self.ce_loss(logits, targets) 