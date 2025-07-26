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
        
        # Нормализуем logits
        logits_norm = F.normalize(logits, p=2, dim=1)
        
        # Вычисляем косинус угла
        cos_theta = torch.clamp(logits_norm, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Вычисляем угол
        theta = torch.acos(cos_theta)
        
        # Создаем маску для целевого класса
        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, labels.unsqueeze(1), 1)
        
        # Применяем margin только к целевому классу
        cos_theta_m = torch.where(mask == 1, 
                                 torch.cos(self.margin * theta), 
                                 cos_theta)
        
        # Масштабируем
        cos_theta_m = cos_theta_m * self.scale
        
        # Вычисляем loss
        loss = F.cross_entropy(cos_theta_m, labels)
        
        return {"loss": loss}