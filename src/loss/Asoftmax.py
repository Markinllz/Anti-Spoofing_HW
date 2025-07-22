import torch
from torch import nn
import torch.nn.functional as F

class AsoftMax(nn.Module):

    def __init__(self, margin=4, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        A-Softmax loss calculation
        
        Args:
            logits (Tensor): model output predictions (batch_size, num_classes)
            labels (Tensor): ground-truth labels (batch_size,)
        Returns:
            losses (dict): dictionary loss
        """
       
        logits_norm = F.normalize(logits, p=2, dim=1)
        
       
        cos_theta = torch.clamp(logits_norm, -1.0 + 1e-8, 1.0 - 1e-8)
        
       
        theta = torch.acos(cos_theta)
        
       
        cos_theta_m = cos_theta.clone()
        
        
        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, labels.unsqueeze(1), 1)
        
       
        cos_theta_m = torch.where(mask == 1, 
                                 torch.cos(self.margin * theta), 
                                 cos_theta)
        
       
        cos_theta_m = cos_theta_m * self.scale
        
   
        loss = F.cross_entropy(cos_theta_m, labels)
        
        return {"loss": loss}