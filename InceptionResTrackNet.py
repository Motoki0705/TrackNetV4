import torch
import torch.nn as nn
import  time

class InceptionResNetBlock_A(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight
        
        # Path 1
        self.path1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        )
        
        # Path 3
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        )
        
        # Output
        self.conv11_out = nn.Conv2d(out_channels * 3, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        
        x_out = torch.cat([x_1, x_2, x_3], dim=1)
        x_out = self.conv11_out(x_out)
        out = self.bn(x_out + x * self.res_weight)
        out = self.relu(out)
        return out

class InceptionResNetBlock_B(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight
        
        # Path 1
        self.path1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), stride=1, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), stride=1, padding='same')
        )
        
        # Output
        self.conv11_out = nn.Conv2d(out_channels * 2, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        
        x_out = torch.cat([x_1, x_2], dim=1)
        x_out = self.conv11_out(x_out)
        out = self.bn(x_out + x * self.res_weight)
        out = self.relu(out)
        return out

class InceptionResNetBlock_C(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight
        
        # Path 1
        self.path1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding='same')
        )
        
        # Output
        self.conv11_out = nn.Conv2d(out_channels * 2, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        
        x_out = torch.cat([x_1, x_2], dim=1)
        x_out = self.conv11_out(x_out)
        out = self.bn(x_out + x * self.res_weight)
        out = self.relu(out)
        return out

class Reduction_A(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight

        # Path 1
        self.path1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        
        # Path 3
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        
        # Output
        self.conv11_out = nn.Conv2d(in_channels + out_channels * 2, in_channels * 2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        
        x_out = torch.cat([x_1, x_2, x_3], dim=1)
        x_out = self.conv11_out(x_out)
        x_shortcut = self.shortcut(x)
        
        out = self.bn(x_out + x_shortcut * self.res_weight)
        out = self.relu(out)
        return out

class Reduction_B(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight

        # Path 1
        self.path1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        
        # Path 3
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        
        # Path 4
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        
        # Output
        self.conv11_out = nn.Conv2d(in_channels + out_channels * 3, in_channels * 2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        x_4 = self.path4(x)
        
        x_out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        x_out = self.conv11_out(x_out)
        x_shortcut = self.shortcut(x)
        
        out = self.bn(x_out + x_shortcut * self.res_weight)
        out = self.relu(out)
        return out

class Expansion_A(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight

        # Path 1: Upsampling
        self.path1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Path 3
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Output
        self.conv11_out = nn.Conv2d(in_channels + out_channels * 2, int(in_channels / 2), kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(int(in_channels / 2))
        self.relu = nn.ReLU()
        self.shortcut = nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        
        x_out = torch.cat([x_1, x_2, x_3], dim=1)
        x_out = self.conv11_out(x_out)
        x_shortcut = self.shortcut(x)
        
        out = self.bn(x_out + x_shortcut * self.res_weight)
        out = self.relu(out)
        return out

class Expansion_B(nn.Module):
    def __init__(self, in_channels, out_channels, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_weight = res_weight

        # Path 1: Upsampling
        self.path1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Path 2
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Path 3
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Path 4
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        # Output
        self.conv11_out = nn.Conv2d(in_channels + out_channels * 3, int(in_channels / 2), kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(int(in_channels / 2))
        self.relu = nn.ReLU()
        self.shortcut = nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        x_4 = self.path4(x)
        
        x_out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        x_out = self.conv11_out(x_out)
        x_shortcut = self.shortcut(x)
        
        out = self.bn(x_out + x_shortcut * self.res_weight)
        out = self.relu(out)
        return out

class InceptionResTrackNet(nn.Module):
    def __init__(self, I, num_A, num_B, num_C, res_weight, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.res_weight = res_weight 
        
        self.resA_1 = nn.Sequential(
            *[InceptionResNetBlock_A(in_channels=I, out_channels=int(I / 4), res_weight=0.1) for _ in range(num_A)]
        )
        
        self.redA_1 = Reduction_A(in_channels=I, out_channels=int(I / 4), res_weight=0.1)
        
        self.resB_1 = nn.Sequential(
            *[InceptionResNetBlock_B(in_channels=I * 2, out_channels=int(I * 2 / 4), res_weight=0.1) for _ in range(num_B)]
        )
        
        self.redB_1 = Reduction_B(in_channels=I * 2, out_channels=int(I * 2 / 4), res_weight=0.1)
        
        self.resC_1 = nn.Sequential(
            *[InceptionResNetBlock_C(in_channels=I * 4, out_channels=int(I * 4 / 4), res_weight=0.1) for _ in range(num_C)]
        )
        
        self.dropout = nn.Dropout2d(p=0.3)
        
        self.resC_2 = nn.Sequential(
            *[InceptionResNetBlock_C(in_channels=I * 4, out_channels=int(I * 4 / 4), res_weight=0.1) for _ in range(num_C)]
        )
        
        self.expB_1 = Expansion_B(in_channels=I * 4, out_channels=int(I * 4 / 4), res_weight=0.1)
        
        self.resB_2 = nn.Sequential(
            *[InceptionResNetBlock_B(in_channels=I * 2, out_channels=int(I * 2 / 4), res_weight=0.1) for _ in range(num_C)]
        )
        
        self.expA_1 = Expansion_A(in_channels=I * 2, out_channels=int(I * 2 / 4), res_weight=0.1)
        
        self.resA_2 = nn.Sequential(
            *[InceptionResNetBlock_A(in_channels=I, out_channels=int(I / 4), res_weight=0.1) for _ in range(num_A)]
        )
        
        self.conv = nn.Conv2d(in_channels=I, out_channels=1, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.resA_1(x)
        x_1 = self.redA_1(x)
        x = self.resB_1(x_1)
        x_2 = self.redB_1(x)
        x = self.resC_1(x_2)
        x = self.dropout(x)
        x = self.resC_2(x)
        x = self.expB_1(x + x_2 * self.res_weight)
        x = self.resB_2(x)
        x = self.expA_1(x + x_1 * self.res_weight)
        x = self.resA_2(x)
        x = self.conv(x)

        x_soft = self.softmax(x.view(x.size(0), -1))
        
        if self.training:
            return x_soft

        else:
            x_maxval, x_index = torch.max(x_soft, dim=1)
            x_coord = x_index % x.size(2)
            y_coord = x_index // x.size(2)
            
            return  x_maxval, (x_coord, y_coord)

if __name__ == '__main__':
    model = InceptionResTrackNet(I=9, num_A=2, num_B=3, num_C=1, res_weight=0.1)
    model.train()
    inputs = torch.randn(1, 9, 180, 360)  # バッチサイズ1、9チャンネル、(180, 360)画像
    
    start = time.time()
    for _ in range(30):
        outputs = model(inputs)
        print(outputs)

    end = time.time()
    print(end - start)