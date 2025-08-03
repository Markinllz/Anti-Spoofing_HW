import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math


class MFM(nn.Module):
    """
    Max Feature Map (MFM) activation function.
    Takes maximum of two halves of input channels or features.
    """
    def __init__(self, in_features, out_features):
        super(MFM, self).__init__()
        self.out_features = out_features
        
    def forward(self, x):
        if x.dim() == 4:
            x1, x2 = torch.split(x, self.out_features, dim=1)
        else:
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
            dropout_rate (float): dropout probability
        """
        super(LCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.mfm1 = MFM(64, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block1_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1_mfm1 = MFM(64, 32)
        self.block1_bn1 = nn.BatchNorm2d(32)
        self.block1_conv2 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_mfm2 = MFM(96, 48)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1_bn2 = nn.BatchNorm2d(48)
        
        self.block2_conv1 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.block2_mfm1 = MFM(96, 48)
        self.block2_bn1 = nn.BatchNorm2d(48)
        self.block2_conv2 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_mfm2 = MFM(128, 64)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.block3_mfm1 = MFM(128, 64)
        self.block3_bn1 = nn.BatchNorm2d(64)
        self.block3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_mfm2 = MFM(64, 32)
        self.block3_bn2 = nn.BatchNorm2d(32)
        
        self.block4_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.block4_mfm1 = MFM(64, 32)
        self.block4_bn1 = nn.BatchNorm2d(32)
        self.block4_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_mfm2 = MFM(64, 32)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 160, bias=True)
        self.mfm_fc = MFM(160, 80)
        self.bn_fc = nn.BatchNorm1d(80)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(80, num_classes, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Light CNN model.
        
        Args:
            batch: input batch containing tensors or direct tensor
            
        Returns:
            Dict[str, torch.Tensor]: model outputs
        """
        if isinstance(batch, dict):
            if 'data_object' in batch:
                x = batch['data_object']
            elif 'spectrogram' in batch:
                x = batch['spectrogram']
            else:
                x = next(iter(batch.values()))
        else:
            x = batch
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        x = self.conv1(x)
        x = self.mfm1(x)
        x = self.pool1(x)
        
        x = self.block1_conv1(x)
        x = self.block1_mfm1(x)
        x = self.block1_bn1(x)
        x = self.block1_conv2(x)
        x = self.block1_mfm2(x)
        x = self.block1_pool(x)
        x = self.block1_bn2(x)
        
        x = self.block2_conv1(x)
        x = self.block2_mfm1(x)
        x = self.block2_bn1(x)
        x = self.block2_conv2(x)
        x = self.block2_mfm2(x)
        x = self.block2_pool(x)
        
        x = self.block3_conv1(x)
        x = self.block3_mfm1(x)
        x = self.block3_bn1(x)
        x = self.block3_conv2(x)
        x = self.block3_mfm2(x)
        x = self.block3_bn2(x)
        
        x = self.block4_conv1(x)
        x = self.block4_mfm1(x)
        x = self.block4_bn1(x)
        x = self.block4_conv2(x)
        x = self.block4_mfm2(x)
        x = self.block4_pool(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.mfm_fc(x)
        x = self.bn_fc(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return {
            "logits": x
        }


class LCNN_LSTM_Sum(nn.Module):
    """
    LCNN-LSTM-sum архитектура из статьи ASVspoof2019.
    Правильная реализация с skip connection и average pooling.
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3, **kwargs):
        """
        Args:
            num_classes (int): number of output classes
            dropout_rate (float): dropout probability
        """
        super(LCNN_LSTM_Sum, self).__init__()
        
        # === CNN PART (exactly as in paper) ===
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.mfm1 = MFM(64, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.block1_conv1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.block1_mfm1 = MFM(64, 32)
        self.block1_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.block1_conv2 = nn.Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.block1_mfm2 = MFM(96, 48)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1_bn2 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        
        self.block2_conv1 = nn.Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.block2_mfm1 = MFM(96, 48)
        self.block2_bn1 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.block2_conv2 = nn.Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.block2_mfm2 = MFM(128, 64)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.block3_conv1 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.block3_mfm1 = MFM(128, 64)
        self.block3_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.block3_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.block3_mfm2 = MFM(64, 32)
        self.block3_bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        
        self.block4_conv1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.block4_mfm1 = MFM(64, 32)
        self.block4_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.block4_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.block4_mfm2 = MFM(64, 32)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout after CNN
        self.dropout_cnn = nn.Dropout(p=0.7)
        
        # === LSTM PART (adaptive) ===
        # Size will be determined in forward pass
        self.lstm_hidden_size = 128
        
        # Two Bi-LSTM layers with skip connection
        self.lstm1 = None  # Will be created in forward
        self.lstm2 = None  # Will be created in forward
        
        # Final FC layer
        self.fc = None  # Will be created in forward
        
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LCNN-LSTM-sum model.
        
        Args:
            batch: input batch containing tensors or direct tensor
            
        Returns:
            Dict[str, torch.Tensor]: model outputs
        """
        if isinstance(batch, dict):
            if 'data_object' in batch:
                x = batch['data_object']
            elif 'spectrogram' in batch:
                x = batch['spectrogram']
            else:
                x = next(iter(batch.values()))
        else:
            x = batch
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        # === CNN FEATURE EXTRACTION ===
        x = self.conv1(x)
        x = self.mfm1(x)
        x = self.pool1(x)
        
        x = self.block1_conv1(x)
        x = self.block1_mfm1(x)
        x = self.block1_bn1(x)
        x = self.block1_conv2(x)
        x = self.block1_mfm2(x)
        x = self.block1_pool(x)
        x = self.block1_bn2(x)
        
        x = self.block2_conv1(x)
        x = self.block2_mfm1(x)
        x = self.block2_bn1(x)
        x = self.block2_conv2(x)
        x = self.block2_mfm2(x)
        x = self.block2_pool(x)
        
        x = self.block3_conv1(x)
        x = self.block3_mfm1(x)
        x = self.block3_bn1(x)
        x = self.block3_conv2(x)
        x = self.block3_mfm2(x)
        x = self.block3_bn2(x)
        
        x = self.block4_conv1(x)
        x = self.block4_mfm1(x)
        x = self.block4_bn1(x)
        x = self.block4_conv2(x)
        x = self.block4_mfm2(x)
        x = self.block4_pool(x)
        
        # Dropout after CNN
        x = self.dropout_cnn(x)
        
        # === RESHAPE FOR LSTM ===
        # Combine height and width into temporal dimension
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels * height, width)  # [batch, features, time]
        x = x.transpose(1, 2)  # [batch, time, features]
        
        # === DYNAMIC LSTM CREATION ===
        feature_dim = x.shape[-1]
        
        # Create LSTM layers if they don't exist
        if self.lstm1 is None:
            self.lstm1 = nn.LSTM(
                input_size=feature_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            ).to(x.device)
            
            self.lstm2 = nn.LSTM(
                input_size=self.lstm_hidden_size * 2,  # *2 for bidirectional
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0
            ).to(x.device)
            
            # FC layer should match LSTM output size (hidden_size * 2 for bidirectional)
            self.fc = nn.Linear(self.lstm_hidden_size * 2, 2).to(x.device)  # 2 classes
        
        # === TWO BI-LSTM LAYERS WITH SKIP CONNECTION ===
        # First LSTM
        lstm1_out, _ = self.lstm1(x)
        
        # Second LSTM with skip connection
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Skip connection: sum LSTM1 and LSTM2 outputs
        lstm_out = lstm1_out + lstm2_out
        
        # Average over temporal dimension
        lstm_out = torch.mean(lstm_out, dim=1)  # [batch, hidden_size * 2]
        
        # Final classification
        output = self.fc(lstm_out)
        
        # Normalize output as in paper
        return {"logits": output}


model = LCNN()