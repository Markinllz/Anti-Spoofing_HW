import torch
import torch.nn as nn
    
class Normalize(nn.Module):
    """
    Нормализация тензоров по каналам.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, x):
        """
        Применяет нормализацию.
        
        Args:
            x (torch.Tensor): входной тензор [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: нормализованный тензор
        """
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        return (x - mean) / std