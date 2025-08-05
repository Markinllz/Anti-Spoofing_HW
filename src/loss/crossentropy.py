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

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Compute cross entropy loss.
        
        Args:
            batch: input batch containing logits and labels
            
        Returns:
            Dict[str, torch.Tensor]: loss dictionary
        """
        # Get logits and labels
        logits = batch['logits']
        labels = batch['labels']
        
        # Check that dimensions are correct for CrossEntropy
        # logits: [batch_size, num_classes], labels: [batch_size]
        assert logits.dim() == 2, f"Expected logits dim=2, got {logits.dim()}"
        assert labels.dim() == 1, f"Expected labels dim=1, got {labels.dim()}"
        assert logits.size(0) == labels.size(0), "Batch size mismatch"
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        return {
            'loss': loss
        }
