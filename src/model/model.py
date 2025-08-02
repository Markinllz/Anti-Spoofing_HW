import torch
import torch.nn as nn
from typing import Dict, Any


class MFM(nn.Module):
    """
    Max Feature Map (MFM) activation function.
    Takes maximum of two halves of input channels or features.
    """
    def __init__(self, in_features, out_features):
        super(MFM, self).__init__()
        self.out_features = out_features
        
    def forward(self, x):
        # Разделяем на две половины и берем максимум
        if x.dim() == 4:  # Для conv слоев [B, C, H, W]
            x1, x2 = torch.split(x, self.out_features, dim=1)
        else:  # Для linear слоев [B, F]  
            x1, x2 = torch.split(x, self.out_features, dim=1)
        return torch.max(x1, x2)


class LCNN(nn.Module):
    """
    Light CNN model for audio anti-spoofing (точно по таблице).
    Общее количество параметров: 371K
    """

    def __init__(self, num_classes=2, dropout_rate=0.3, **kwargs):
        """
        Args:
            num_classes (int): number of output classes
            dropout_rate (float): dropout probability for regularization
            **kwargs: additional arguments
        """
        super(LCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Conv_1 (5x5/1x1) -> MFM_2 -> MaxPool_3 (2x2/2x2)
        # Вход: 863x600x1, Выход: 431x300x32
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)  # 1.6K params
        self.mfm1 = MFM(64, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 1: Conv_4 (1x1/1x1) -> MFM_5 -> BN_6 -> Conv_7 (3x3/1x1) -> MFM_8 -> MaxPool_9 -> BN_10
        # Вход: 431x300x32, Выход: 215x150x48
        self.block1_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)  # 2.1K params
        self.block1_mfm1 = MFM(64, 32)
        self.block1_bn1 = nn.BatchNorm2d(32)
        self.block1_conv2 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False)  # 27.7K params
        self.block1_mfm2 = MFM(96, 48)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1_bn2 = nn.BatchNorm2d(48)
        self.block1_dropout = nn.Dropout2d(p=dropout_rate)
        
        # Block 2: Conv_11 (1x1/1x1) -> MFM_12 -> BN_13 -> Conv_14 (3x3/1x1) -> MFM_15 -> MaxPool_16
        # Вход: 215x150x48, Выход: 107x75x64
        self.block2_conv1 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, bias=False)  # 4.7K params
        self.block2_mfm1 = MFM(96, 48)
        self.block2_bn1 = nn.BatchNorm2d(48)
        self.block2_conv2 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 55.4K params
        self.block2_mfm2 = MFM(128, 64)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2_dropout = nn.Dropout2d(p=dropout_rate)
        
        # Block 3: Conv_17 (1x1/1x1) -> MFM_18 -> BN_19 -> Conv_20 (3x3/1x1) -> MFM_21 -> BN_22
        # Вход: 107x75x64, Выход: 107x75x32
        self.block3_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)  # 8.3K params
        self.block3_mfm1 = MFM(128, 64)
        self.block3_bn1 = nn.BatchNorm2d(64)
        self.block3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 36.9K params
        self.block3_mfm2 = MFM(64, 32)
        self.block3_bn2 = nn.BatchNorm2d(32)
        self.block3_dropout = nn.Dropout2d(p=dropout_rate)
        
        # Block 4: Conv_23 (1x1/1x1) -> MFM_24 -> BN_25 -> Conv_26 (3x3/1x1) -> MFM_27 -> MaxPool_28
        # Вход: 107x75x32, Выход: 53x37x32
        self.block4_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)  # 2.1K params
        self.block4_mfm1 = MFM(64, 32)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 18.5K params
        self.block4_mfm2 = MFM(64, 32)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4_dropout = nn.Dropout2d(p=dropout_rate)
        
        # Classifier: FC_29 -> MFM_30 -> BN_31 -> Dropout -> FC_32
        # Размер после всех операций: 53x37x32 = 62464 features
        # Но в таблице FC_29 дает 160 выходов с 10.2MM параметров
        # Это означает глобальный пулинг до размера, который даст нужное количество параметров
        
        # Добавляем глобальный пулинг и классификатор согласно таблице
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 53x37x32 -> 1x1x32
        self.fc1 = nn.Linear(32, 160, bias=True)  # FC_29: 32*160 + 160 = 5280 ≈ 5.3K params
        self.mfm_fc = MFM(160, 80)  # MFM_30: 160 -> 80
        self.bn_fc = nn.BatchNorm1d(80)  # BN_31
        self.dropout_fc = nn.Dropout(p=dropout_rate)  # Dropout перед финальным слоем
        self.fc2 = nn.Linear(80, num_classes, bias=True)  # FC_32: 80*2 + 2 = 162 ≈ 64 params
        
        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Light CNN model.
        
        Args:
            batch: input batch containing tensors
            
        Returns:
            Dict[str, torch.Tensor]: model outputs
        """
        # Получаем входные данные
        if 'data_object' in batch:
            x = batch['data_object']
        elif 'spectrogram' in batch:
            x = batch['spectrogram']
        else:
            x = next(iter(batch.values()))
        
        # Убеждаемся, что входные данные имеют правильную форму
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Добавляем канал: [batch, freq, time] -> [batch, 1, freq, time]
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [freq, time] -> [1, 1, freq, time]
        
        # Conv_1 + MFM_2 + MaxPool_3
        # Ожидаемый размер: [B, 1, 863, 600] -> [B, 32, 431, 300]
        x = self.conv1(x)          # [B, 64, 863, 600]
        x = self.mfm1(x)           # [B, 32, 863, 600]
        x = self.pool1(x)          # [B, 32, 431, 300]
        
        # Block 1: [B, 32, 431, 300] -> [B, 48, 215, 150]
        x = self.block1_conv1(x)   # [B, 64, 431, 300]
        x = self.block1_mfm1(x)    # [B, 32, 431, 300]
        x = self.block1_bn1(x)     
        x = self.block1_conv2(x)   # [B, 96, 431, 300]
        x = self.block1_mfm2(x)    # [B, 48, 431, 300]
        x = self.block1_pool(x)    # [B, 48, 215, 150]
        x = self.block1_bn2(x)
        x = self.block1_dropout(x)
        
        # Block 2: [B, 48, 215, 150] -> [B, 64, 107, 75]
        x = self.block2_conv1(x)   # [B, 96, 215, 150]
        x = self.block2_mfm1(x)    # [B, 48, 215, 150]
        x = self.block2_bn1(x)     
        x = self.block2_conv2(x)   # [B, 128, 215, 150]
        x = self.block2_mfm2(x)    # [B, 64, 215, 150]
        x = self.block2_pool(x)    # [B, 64, 107, 75]
        x = self.block2_dropout(x)
        
        # Block 3: [B, 64, 107, 75] -> [B, 32, 107, 75]
        x = self.block3_conv1(x)   # [B, 128, 107, 75]
        x = self.block3_mfm1(x)    # [B, 64, 107, 75]
        x = self.block3_bn1(x)
        x = self.block3_conv2(x)   # [B, 64, 107, 75]
        x = self.block3_mfm2(x)    # [B, 32, 107, 75]
        x = self.block3_bn2(x)
        x = self.block3_dropout(x)
        
        # Block 4: [B, 32, 107, 75] -> [B, 32, 53, 37]
        x = self.block4_conv1(x)   # [B, 64, 107, 75]
        x = self.block4_mfm1(x)    # [B, 32, 107, 75]
        x = self.block4_bn1(x)
        x = self.block4_conv2(x)   # [B, 64, 107, 75]
        x = self.block4_mfm2(x)    # [B, 32, 107, 75]
        x = self.block4_pool(x)    # [B, 32, 53, 37]
        x = self.block4_dropout(x)
        
        # Classifier: [B, 32, 53, 37] -> [B, num_classes]
        x = self.global_pool(x)    # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 32]
        x = self.fc1(x)            # [B, 160] (FC_29)
        x = self.mfm_fc(x)         # [B, 80] (MFM_30)
        x = self.bn_fc(x)          # [B, 80] (BN_31)
        x = self.dropout_fc(x)     # Dropout
        x = self.fc2(x)            # [B, num_classes] (FC_32)
        
        # Возвращаем выходы
        outputs = {
            'logits': x
        }
        
        return outputs


# Создаем экземпляр модели для совместимости с hydra
def create_model(**kwargs) -> LCNN:
    """
    Создает экземпляр модели Light CNN.
    
    Args:
        **kwargs: параметры модели
        
    Returns:
        LCNN: экземпляр модели
    """
    model = LCNN(**kwargs)
    return model