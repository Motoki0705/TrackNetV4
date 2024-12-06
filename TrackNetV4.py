import torch
import torch.nn as nn
import torch.nn.functional as F
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)  # [B, C, 1, 1]
        x = x * ca

        # Spatial Attention
        sa = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)  # [B, 2, H, W]
        sa = self.spatial_attention(sa)  # [B, 1, H, W]
        x = x * sa
        return x
      
class ResidualDownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(ResidualDownSamplingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ショートカット
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # CBAMはオプション
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_cbam:
            out = self.cbam(out)
        
        out += identity
        return self.relu(out)
 
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        # Apply upsampling
        return self.upsample(x)
 
class TrackNetV4(nn.Module):
    def __init__(self, in_chanels, out_chanels, num_layers, buffer_size):
        super(TrackNetV4, self).__init__()
        self.out_chanels = out_chanels
        self.buffer_size = buffer_size

        # ダウンサンプリング
        self.downblock_1 = ResidualDownSamplingBlock(in_chanels, 64, stride=2, use_cbam=True)
        self.downblock_2 = ResidualDownSamplingBlock(64, 128, stride=2, use_cbam=True)
        self.downblock_3 = ResidualDownSamplingBlock(128, 256, stride=2, use_cbam=True)
        self.downblock_4 = ResidualDownSamplingBlock(256, 512, stride=2, use_cbam=True)
        self.downblock_5 = ResidualDownSamplingBlock(512, out_chanels, stride=2, use_cbam=True)

        # フィーチャー処理
        self.flatten = nn.Flatten()
        self.fc_down = nn.Sequential(
            nn.Linear(out_chanels * 10 * 20, out_chanels * 10),
            nn.BatchNorm1d(out_chanels * 10),
            nn.ReLU(),
            nn.Linear(out_chanels * 10, out_chanels),
            nn.BatchNorm1d(out_chanels),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(out_chanels, out_chanels, num_layers, batch_first=True)

        # フィーチャー処理
        self.fc_up = nn.Sequential(
            nn.Linear(out_chanels, out_chanels * 10),
            nn.BatchNorm1d(out_chanels * 10),
            nn.ReLU(),
            nn.Linear(out_chanels * 10, out_chanels * 10 * 20),
            nn.BatchNorm1d(out_chanels * 10 * 20),
            nn.ReLU()
        )

        # アップサンプリング
        self.upblock_1 = UpsampleBlock(out_chanels, 512, kernel_size=2, stride=2)
        self.upblock_2 = UpsampleBlock(512, 256, kernel_size=2, stride=2)
        self.upblock_3 = UpsampleBlock(256, 128, kernel_size=2, stride=2)
        self.upblock_4 = UpsampleBlock(128, 64, kernel_size=2, stride=2)
        self.upblock_5 = UpsampleBlock(64, 1, kernel_size=2, stride=2)

        # バッファ
        self.register_buffer("feature_buffer", torch.empty(0, out_chanels))

    def forward(self, x):
        x_1 = self.downblock_1(x)
        x_2 = self.downblock_2(x_1)
        x_3 = self.downblock_3(x_2)
        x_4 = self.downblock_4(x_3)
        x_5 = self.downblock_5(x_4)

        # フィーチャー処理
        x = self.flatten(x_5)
        x = self.fc_down(x)

        # バッファに追加
        self.feature_buffer = torch.cat([self.feature_buffer, x], dim=0)

        if self.feature_buffer.size(0) == self.buffer_size:
            lstm_out, _ = self.lstm(self.feature_buffer.unsqueeze(0))
            lstm_out = lstm_out[:, -1, :]

            # アップサンプリング
            x = self.fc_up(lstm_out)
            x = x.view(1, self.out_chanels, 10, 20)
            x = self.upblock_1(x + x_5)
            x = self.upblock_2(x + x_4)
            x = self.upblock_3(x + x_3)
            x = self.upblock_4(x + x_2)
            x = self.upblock_5(x + x_1)

            # ソフトマックスを適用
            x_softmaxed = F.softmax(x.view(1, -1), dim=1)

            # 最大値とその座標を取得
            max_value, max_index = torch.max(x_softmaxed.view(-1), dim=0)
            
            coords = torch.div(max_index, x.size(3), rounding_mode='floor'), max_index % x.size(3)

            # バッファを更新
            self.feature_buffer = self.feature_buffer[1:]

            return max_value.item(), coords
        else:
            return None
            
   
if __name__ == '__main__':         
    model = TrackNetV4(in_chanels=3, out_chanels=512, num_layers=3, buffer_size=3)
    model.eval()

    for i in range(25):
        inputs = torch.randn(1, 3, 320, 640)
        result = model(inputs)
        if result is not None:
            print(result)
        else:
            print('buffer is not enough')
            