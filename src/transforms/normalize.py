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
        
        # Адаптируем размерности mean и std под входной тензор
        if x.dim() == 4:  # [batch, channels, height, width]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif x.dim() == 3:  # [batch, freq, time] - наш случай после STFT
            mean = mean.view(1, 1, 1) if len(mean) == 1 else mean.view(1, -1, 1)
            std = std.view(1, 1, 1) if len(std) == 1 else std.view(1, -1, 1)
        elif x.dim() == 2:  # [batch, features]
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        
        return (x - mean) / std