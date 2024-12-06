import os
import cv2
import re
import pandas as pd
from CreateLabel import create_labels_with_heatmap_optimized
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

# 親ディレクトリ内のサブディレクトリを取得する関数
def get_subdirectories(parent_directory):
    """指定されたディレクトリ内のサブディレクトリを取得"""
    entries = os.listdir(parent_directory)
    subdirectories = []
    
    for entry in entries:
        full_path = os.path.join(parent_directory, entry)
        if os.path.isdir(full_path):
            subdirectories.append(full_path)
            
    return subdirectories

# 指定されたディレクトリ内の .jpg ファイルを取得する関数
def get_jpg_files(directory):
    """指定されたディレクトリ内の .jpg ファイルのパスを取得"""
    jpg_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    return jpg_files

def get_csv_files(directory):
    """指定されたディレクトリ内の .csv ファイルのパスを取得"""
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    return csv_files

# ファイルパスを数値部分でソートする関数
def sort_image_paths(image_paths):
    """画像ファイルのパスを数値部分でソート"""
    sorted_paths = sorted(image_paths, key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
    return sorted_paths

def sort_game_clip_dirs(directories):
    sorted_directories = sorted(directories, key=lambda dir: int(re.search(r'\d+', os.path.basename(dir)).group()))
    return sorted_directories

# データセット内のすべての画像ファイルのフルパスを取得する関数
def collect_dataset_image_and_label_paths(dataset_directory):
    """データセット内の画像ファイルのフルパスをすべて収集"""
    game_directories = get_subdirectories(dataset_directory)
    game_directories = sort_game_clip_dirs(game_directories)
    all_game_image_paths = []
    all_game_label_paths = []
    
    for game_directory in game_directories:
        clip_directories = get_subdirectories(game_directory)
        clip_directories = sort_game_clip_dirs(clip_directories)
        game_image_paths = []
        game_label_paths = []
        
        for clip_directory in clip_directories:
            image_paths = get_jpg_files(clip_directory)
            label_path = get_csv_files(clip_directory)
            sorted_image_paths = sort_image_paths(image_paths)
            game_image_paths.append(sorted_image_paths)
            game_label_paths.append(label_path)
            
        all_game_image_paths.append(game_image_paths)
        all_game_label_paths.append(game_label_paths)
        
    return all_game_image_paths, all_game_label_paths


def create_dataset(image_paths, label_paths, num_series, sigma=5):
    dataset = []
    for game_idx, game_level in enumerate(image_paths):
        for clip_idx, clip_level in enumerate(game_level):
            label_path = label_paths[game_idx][clip_idx]
            
            df = pd.read_csv(label_path[0])
            heatmap_labels = create_labels_with_heatmap_optimized(df, sigma=sigma)
            
            for i in range(len(clip_level) - num_series + 1):
                series_paths = clip_level[i:i + num_series]
                label = heatmap_labels[i + num_series - 1]
                dataset.append([series_paths, label])
    
    return dataset

# PyTorch Datasetクラス
class CustomDataset(Dataset):
    def __init__(self, dataset_generator, transform=None):
        """
        Args:
            dataset_generator (generator): データセットジェネレータ。
            transform (callable, optional): 画像に適用する変換処理。
        """
        self.dataset = list(dataset_generator)
        self.transform = transform

    def __len__(self):
        """データセットのサイズを返す"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        指定されたインデックスに基づき、データとラベルを返す。
        Returns:
            images (torch.Tensor): 画像テンソル ([num_series * C, H, W])。
            label (torch.Tensor): ラベルテンソル。
        """
        image_paths, label = self.dataset[idx]
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")  # 画像をRGB形式で読み込む
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # チャンネル方向に結合 ([num_series * C, H, W])
        images = torch.cat(images, dim=0)
        return images, torch.tensor(label)

# メイン処理
if __name__ == '__main__':
    dataset_directory = r"C:\Users\kamim\Downloads\Dataset\Dataset"
    num_series = 3  # シリーズ長

    # 画像とラベルのパスを収集
    image_paths, label_paths = collect_dataset_image_and_label_paths(dataset_directory)

    # 画像変換（リサイズと正規化）
    transform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor()
    ])

    # データセットジェネレータを作成
    dataset_generator = create_dataset(image_paths, label_paths, num_series, sigma=10)

    # PyTorch Datasetを作成
    custom_dataset = CustomDataset(dataset_generator, transform=transform)

    # DataLoaderの使用
    data_loader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

    # データの確認
    for images, labels in data_loader:
        print(images.shape)  # 出力: torch.Size([8, 9, 360, 640])
        print(labels.shape)  # 出力: torch.Size([8, 360 * 640])
        break
