import torch
import torch.nn as nn
from typing import Dict, Any


class LCNN(nn.Module):
    """
    Light CNN model for audio anti-spoofing.
    """

    def __init__(self, num_classes=2, **kwargs):
        """
        Args:
            num_classes (int): number of output classes
            **kwargs: additional arguments
        """
        super(LCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Определяем архитектуру
        self.features = nn.Sequential(
            # Первый блок
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Второй блок
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Третий блок
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Четвертый блок
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # ДОБАВЛЯЕМ ПРАВИЛЬНУЮ ИНИЦИАЛИЗАЦИЮ ВЕСОВ
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            **batch: input batch containing tensors
            
        Returns:
            Dict[str, torch.Tensor]: model outputs
        """
        # Получаем входные данные
        if 'spectrogram' in batch:
            x = batch['spectrogram']
        elif 'data_object' in batch:
            x = batch['data_object']
        else:
            # Берем первый тензор из батча
            x = next(iter(batch.values()))
        
        # Убеждаемся, что входные данные имеют правильную форму
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Добавляем канал
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Добавляем batch и канал
        
        # Проходим через слои
        x = self.features(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        # Возвращаем выходы
        outputs = {
            'logits': x
        }
        
        return outputs


# Создаем экземпляр модели для совместимости с hydra
def create_model(**kwargs) -> LCNN:
    """
    Создает экземпляр модели LCNN.
    
    Args:
        **kwargs: параметры модели
        
    Returns:
        LCNN: экземпляр модели
    """
    model = LCNN(**kwargs)
    return model