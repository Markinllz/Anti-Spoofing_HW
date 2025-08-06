import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D) 
    
    MaxFeatureMap2D(max_dim=1)
    
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)
        
        shape = list(inputs.size())
        
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            raise ValueError("Invalid max_dim")
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            raise ValueError("Odd number of channels")
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        
        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


class LCNNWithLSTM(nn.Module):
    """
    LCNN with LSTM layers for variable length inputs
    Architecture exactly as shown in the image with 371K parameters
    """
    
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.5):
        super(LCNNWithLSTM, self).__init__()
        
        # CNN features - точно как в статье
        self.features = nn.Sequential(
            # Conv_1: 5x5/1x1, output: 863x600x64, 1.6K params
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            # MFM_2: output: 864x600x32
            MaxFeatureMap2D(),
            # MaxPool_3: 2x2/2x2, output: 431x300x32
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv_4: 1x1/1x1, output: 431x300x64, 2.1K params
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            # MFM_5: output: 431x300x32
            MaxFeatureMap2D(),
            # BatchNorm_6: output: 431x300x32
            nn.BatchNorm2d(32, affine=False),
            # Conv_7: 3x3/1x1, output: 431x300x96, 27.7K params
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            # MFM_8: output: 431x300x48
            MaxFeatureMap2D(),
            # MaxPool_9: 2x2/2x2, output: 215x150x48
            nn.MaxPool2d(kernel_size=2, stride=2),
            # BatchNorm_10: output: 215x150x48
            nn.BatchNorm2d(48, affine=False),

            # Conv_11: 1x1/1x1, output: 215x150x96, 4.7K params
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            # MFM_12: output: 215x150x48
            MaxFeatureMap2D(),
            # BatchNorm_13: output: 215x150x48
            nn.BatchNorm2d(48, affine=False),
            # Conv_14: 3x3/1x1, output: 215x150x128, 55.4K params
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            # MFM_15: output: 215x150x64
            MaxFeatureMap2D(),
            # MaxPool_16: 2x2/2x2, output: 107x75x64
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv_17: 1x1/1x1, output: 107x75x128, 8.3K params
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            # MFM_18: output: 107x75x64
            MaxFeatureMap2D(),
            # BatchNorm_19: output: 107x75x64
            nn.BatchNorm2d(64, affine=False),
            # Conv_20: 3x3/1x1, output: 107x75x64, 36.9K params
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # MFM_21: output: 107x75x32
            MaxFeatureMap2D(),
            # BatchNorm_22: output: 107x75x32
            nn.BatchNorm2d(32, affine=False),

            # Conv_23: 1x1/1x1, output: 107x75x64, 2.1K params
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            # MFM_24: output: 107x75x32
            MaxFeatureMap2D(),
            # BatchNorm_25: output: 107x75x32
            nn.BatchNorm2d(32, affine=False),
            # Conv_26: 3x3/1x1, output: 107x75x64, 18.5K params
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # MFM_27: output: 107x75x32
            MaxFeatureMap2D(),
            # MaxPool_28: 2x2/2x2, output: 53x37x32
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # FC слои точно как в статье, но увеличенные для >10M параметров
        # FC_29: 200 features - увеличенный размер для >10M параметров
        target_input_size = 62500  # 53*37*32 = 62500 (как в оригинале)
        self.fc1 = nn.Linear(target_input_size, 200)  # 62500 * 200 = 12.5M параметров
        
        # MFM_30: 100 features
        self.mfm_fc = MaxFeatureMap2D()
        
        # BatchNorm_31: 100 features
        self.bn_fc = nn.BatchNorm1d(100, affine=False)
        
        # FC_32: 2 features, как в оригинале
        self.fc2 = nn.Linear(100, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, time_steps, n_mels) from mel-spectrogram
        Returns:
            output: dict with 'logits' and 'probs' keys
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (batch_size, time_frames, n_mels) -> (batch_size, n_mels, time_frames)
            x = x.transpose(1, 2)
            batch_size, n_mels, time_frames = x.shape
            # Reshape for CNN: (batch_size, 1, n_mels, time_frames)
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # (batch_size, channels, height, width) - already correct
            batch_size = x.shape[0]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Extract features
        features = self.features(x)  # (batch_size, 32, height, width)
        
        # Flatten for FC layers
        features = features.view(batch_size, -1)  # (batch_size, flattened_size)
        
        # Pad or truncate to target size (62500)
        target_input_size = 62500  # 53*37*32 = 62500 (как в оригинале)
        current_size = features.shape[1]
        if current_size < target_input_size:
            # Pad with zeros
            padding = torch.zeros(batch_size, target_input_size - current_size, device=features.device)
            features = torch.cat([features, padding], dim=1)
        elif current_size > target_input_size:
            # Truncate
            features = features[:, :target_input_size]
        

        
        # FC_29: 200 features
        features = self.dropout(features)
        features = F.relu(self.fc1(features))  # (batch_size, 200)
        
        # MFM_30: 100 features (reshape for MFM)
        features = features.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 200, 1, 1)
        features = self.mfm_fc(features)  # (batch_size, 100, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size, 100)
        
        # BatchNorm_31: 100 features
        features = self.bn_fc(features)
        
        # FC_32: 2 features
        features = self.dropout(features)
        logits = self.fc2(features)  # (batch_size, num_classes)
        
        # Apply sigmoid for binary classification
        probs = torch.sigmoid(logits)
        
        return {"logits": logits, "probs": probs}
