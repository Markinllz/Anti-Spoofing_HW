import torch
import torch.nn as nn
    
class Normalize(nn.Module):
    """
    Нормализация тензоров по каналам.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        """
        Применяет нормализацию.
        
        Args:
            x (torch.Tensor): входной тензор любой размерности
            
        Returns:
            torch.Tensor: нормализованный тензор
        """
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        # Adapt mean and std dimensions to input tensor
        if x.dim() == 4:  # [batch, channels, freq, time]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif x.dim() == 3:  # [batch, freq, time] - our case after STFT
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        else:
            mean = mean
            std = std
        
        return (x - mean) / std