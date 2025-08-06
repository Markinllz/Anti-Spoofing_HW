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
    Simplified for binary classification with sigmoid
    """
    
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.5):
        super(LCNNWithLSTM, self).__init__()
        
        # CNN features
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48, affine=False),

            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),

            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # LSTM layers - adjust input size based on CNN output
        self.lstm1 = nn.LSTM(96, 32, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, bidirectional=True, batch_first=True)
        
        # Average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.output_act = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)  # Single output for binary classification
        )
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, channels, time_steps, feature_dim) or (batch_size, time_steps, feature_dim)
        Returns:
            output: dict with 'logits' and 'probs' keys
        """
        # Handle both 3D and 4D inputs
        if x.dim() == 4:
            # Remove channel dimension if present
            x = x.squeeze(1)  # (batch_size, time_steps, feature_dim)
        
        batch_size, time_steps, feature_dim = x.shape
        
        # Reshape for CNN: (batch_size, 1, time_steps, feature_dim)
        x = x.unsqueeze(1)
        
        # Extract features
        features = self.features(x)  # (batch_size, 32, time_steps//16, feature_dim//16)
        
        # Reshape for LSTM: (batch_size, time_steps//16, features)
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(batch_size, features.size(1), -1)
        
        # LSTM layers
        lstm_out1, _ = self.lstm1(features)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Average pooling
        pooled = self.avg_pool(lstm_out2.transpose(1, 2)).squeeze(-1)
        
        # Output layers
        logits = self.output_act(pooled)
        # Apply sigmoid for binary classification
        probs = torch.sigmoid(logits)
        
        return {"logits": logits, "probs": probs}
