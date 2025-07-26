import torch
from torch import nn
import torch.nn.functional as F

class AsoftMax(nn.Module):

    def __init__(self, margin=4, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        A-Softmax loss compute
        
        Args:
            logits (Tensor): model output predictions (batch_size, num_classes)
            labels (Tensor): ground truth labels (batch_size,)
            **kwargs: дополнительные аргументы (игнорируются)
        Returns:
            losses (dict): dictionary loss
        """
       
        logits_norm = F.normalize(logits, p=2, dim=1)
        prev_cos = torch.clamp(logits_norm, -1.0 + 1e-6, 1.0 - 1e-6)
        
        angle = torch.acos(prev_cos)
        cos_m = prev_cos.clone()
        
        
        mask = torch.zeros_like(prev_cos)
        mask.scatter_(1, labels.unsqueeze(1), 1)
        
       
        cos_m = torch.where(mask == 1, 
                                 torch.cos(self.margin * angle), 
                                 prev_cos)
        
       
        cos_m = cos_m * self.scale
        
   
        loss = F.cross_entropy(cos_m, labels)
        
        return {"loss": loss}