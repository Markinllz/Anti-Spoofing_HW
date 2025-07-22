import torch
from torch import nn

class mfm_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        partition = self.channels // 2
        first_batch = x[:, :partition, ...]
        second_batch = x[:, partition:, ...]
        output = torch.maximum(first_batch, second_batch)
        return output

class LCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dropout_p=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.mfm2 = mfm_block(64)
        self.dropout2 = nn.Dropout2d(dropout_p)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm5 = mfm_block(64)
        self.dropout5 = nn.Dropout2d(dropout_p)
        self.BatchNorm6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm8 = mfm_block(96)
        self.dropout8 = nn.Dropout2d(dropout_p)

        self.MaxPool9 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.BatchNorm10 = nn.BatchNorm2d(48)

        self.conv11 = nn.Conv2d(48, 96, kernel_size=1, stride=1)
        self.mfm12 = mfm_block(96)
        self.dropout12 = nn.Dropout2d(dropout_p)
        self.BatchNorm13 = nn.BatchNorm2d(48)

        self.conv14 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm15 = mfm_block(128)
        self.dropout15 = nn.Dropout2d(dropout_p)

        self.MaxPool16 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv17 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.mfm18 = mfm_block(128)
        self.dropout18 = nn.Dropout2d(dropout_p)
        self.BatchNorm19 = nn.BatchNorm2d(64)

        self.conv20 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm21 = mfm_block(64)
        self.dropout21 = nn.Dropout2d(dropout_p)
        self.BatchNorm22 = nn.BatchNorm2d(32)

        self.conv23 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm24 = mfm_block(64)
        self.dropout24 = nn.Dropout2d(dropout_p)
        self.BatchNorm25 = nn.BatchNorm2d(32)

        self.conv26 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm27 = mfm_block(64)
        self.dropout27 = nn.Dropout2d(dropout_p)

        self.MaxPool28 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc29 = nn.Linear(32, 160)
        self.dropout29 = nn.Dropout(dropout_p)
        self.mfm30 = mfm_block(160)
        self.BatchNorm31 = nn.BatchNorm1d(80)
        self.fc32 = nn.Linear(80, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mfm2(x)
        x = self.dropout2(x)
        x = self.MaxPool3(x)

        x = self.conv4(x)
        x = self.mfm5(x)
        x = self.dropout5(x)
        x = self.BatchNorm6(x)

        x = self.conv7(x)
        x = self.mfm8(x)
        x = self.dropout8(x)

        x = self.MaxPool9(x)
        x = self.BatchNorm10(x)

        x = self.conv11(x)
        x = self.mfm12(x)
        x = self.dropout12(x)
        x = self.BatchNorm13(x)

        x = self.conv14(x)
        x = self.mfm15(x)
        x = self.dropout15(x)

        x = self.MaxPool16(x)

        x = self.conv17(x)
        x = self.mfm18(x)
        x = self.dropout18(x)
        x = self.BatchNorm19(x)

        x = self.conv20(x)
        x = self.mfm21(x)
        x = self.dropout21(x)
        x = self.BatchNorm22(x)

        x = self.conv23(x)
        x = self.mfm24(x)
        x = self.dropout24(x)
        x = self.BatchNorm25(x)

        x = self.conv26(x)
        x = self.mfm27(x)
        x = self.dropout27(x)

        x = self.MaxPool28(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc29(x)
        x = self.dropout29(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.mfm30(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.BatchNorm31(x)
        x = self.fc32(x)
        return x

