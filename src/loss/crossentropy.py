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
        
        print("🎯 Инициализация CrossEntropyLoss...")
        
        # Логируем параметры
        for key, value in kwargs.items():
            print(f"   📊 {key}: {value}")
        
        self.criterion = nn.CrossEntropyLoss()
        print("✅ CrossEntropyLoss инициализирован")

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        """
        Compute cross entropy loss.
        
        Args:
            **batch: input batch containing logits and labels
            
        Returns:
            Dict[str, torch.Tensor]: loss dictionary
        """
        # Логируем входные данные (только для отладки)
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   💔 CrossEntropyLoss forward: входные ключи {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Получаем logits и labels
        logits = batch['logits']
        labels = batch['labels']
        
        # Проверяем размеры
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # Логируем размеры
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   📊 Logits: shape={logits.shape}, range=[{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"   📊 Labels: shape={labels.shape}, unique={torch.unique(labels).tolist()}")
        
        # Вычисляем потерю
        loss = self.criterion(logits, labels)
        
        # Логируем результат
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   💔 Loss: {loss.item():.4f}")
        
        return {
            'loss': loss
        }

    def set_debug_mode(self, debug_forward=False):
        """
        Включает режим отладки для логирования forward pass.
        
        Args:
            debug_forward (bool): логировать forward pass
        """
        self._debug_forward = debug_forward
        if debug_forward:
            print(f"🐛 Режим отладки включен для {self.__class__.__name__}")
            print(f"   💔 Debug forward: {debug_forward}")
