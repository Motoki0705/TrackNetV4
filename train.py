import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TrackNetV4 import TrackNetV4
from CreateDataloder import CustomDataset
from torchvision import transforms
from tqdm import tqdm


# 設定
config = {
    "dataset_directory": r"C:\Users\kamim\Downloads\Dataset\Dataset",
    "input_height": 180,
    "input_width": 360,
    "in_channels": 3,
    "out_channels": 512,
    "lstm_layers": 3,
    "buffer_size": 3,
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 1e-3,
}

# データセット準備
transform = transforms.Compose([
    transforms.Resize((config["input_height"], config["input_width"])),
    transforms.ToTensor()
])
dataset = CustomDataset(dataset_directory=config["dataset_directory"], transform=transform)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# モデル、損失関数、オプティマイザ
model = TrackNetV4(
    in_chanels=config["in_channels"],
    out_chanels=config["out_channels"],
    num_layers=config["lstm_layers"],
    buffer_size=config["buffer_size"]
).to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# トレーニングループ
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train()

for epoch in range(config["epochs"]):
    epoch_loss = 0.0
    for batch in tqdm(dataloader):
        imgs, labels = batch  # imgs: (batch, buffer_size, 3, 180, 360), labels: (batch, 3)
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # 各バッチ内の逐次処理
        batch_loss = 0.0
        for t in range(config["buffer_size"]):
            # t番目のフレームを取り出してモデルに入力
            inputs = imgs[:, t, :, :, :]  # (batch, 3, 180, 360)
            result = model(inputs)

            # 出力がNoneの場合スキップ
            if result is None:
                continue
            
            visibility_pred, coords_pred = result  # 予測された最大値と座標
            visibility_label = labels[:, 0]       # 正解のvisibility
            coords_label = labels[:, 1:]         # 正解の座標 (x, y)

            # 損失計算
            visibility_loss = criterion(torch.tensor([visibility_pred]).to(device), visibility_label)
            coords_loss = criterion(torch.tensor(coords_pred).float().to(device), coords_label)
            batch_loss += visibility_loss + coords_loss

        # バックプロパゲーションと最適化
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {epoch_loss:.4f}")
