import os 
import cv2
import pandas as pd
import numpy as np
import time

def play_video_with_heatmap(image_paths, heatmaps, fps=30):
    """
    画像を順に読み込み、指定した座標に赤点を描画し、ヒートマップを重ねながらリアルタイムで再生する。

    :param image_paths: 画像ファイルのパスリスト
    :param points: 赤点の座標リスト（[(x1, y1), (x2, y2), ...]）
    :param heatmaps: ヒートマップのリスト（画像サイズに対応）
    :param fps: フレームレート（デフォルト30）
    :param dot_color: 赤点の色 (B, G, R)
    :param dot_radius: 赤点の半径
    """
    delay = 1 / fps  # フレーム間の遅延時間（秒）

    for img_path, heatmap in zip(image_paths, heatmaps):
        # 画像を読み込む
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # リサイズ後の幅と高さを指定
        new_width = 640
        new_height = 360

        # リサイズ
        resized_img = cv2.resize(img, (new_width, new_height))
        heatmap_resized = cv2.resize(heatmap, (new_width, new_height))

        # ヒートマップをカラー化
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # ヒートマップと画像を合成
        overlay = cv2.addWeighted(resized_img, 0.6, heatmap_colored, 0.4, 0)

        # 画像を表示
        cv2.imshow("Video with Heatmap", overlay)

        # フレームレートに応じて待機
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):  # 'q' で終了
            break

    # 再生終了後、ウィンドウを閉じる
    cv2.destroyAllWindows()


def create_labels_with_heatmap_optimized(df_label, output_shape=(360, 720), sigma=5, ball_absent_strategy="uniform"):
    """
    ボールが存在しない場合の処理を含めた効率的なヒートマップ生成。

    :param scaled_coordinates: スケールされた座標リスト [(x1, y1), (x2, y2), ...]
    :param output_shape: 出力画像の形状 (height, width)
    :param sigma: ガウス分布の標準偏差
    :param ball_absent_strategy: ボールが存在しない場合の処理 ("zero" または "uniform")
    :return: 平坦化されたヒートマップラベル配列
    """
    x_scale = 640 / 1280
    y_scale = 360 / 720
    df_label.fillna(0, inplace=True)

    # 座標をスケール変換
    scaled_coordinates = [
        (int(row['x-coordinate'] * x_scale), int(row['y-coordinate'] * y_scale))
        for _, row in df_label.iterrows()
    ]
    
    height, width = output_shape
    labels = []

    for coord in scaled_coordinates:
        label = np.zeros((height, width), dtype=np.float32)

        if coord == (0, 0):  # ボールが映っていない場合
            if ball_absent_strategy == "zero":
                # ラベル全体をゼロに設定
                label = np.zeros((height, width), dtype=np.float32)
            elif ball_absent_strategy == "uniform":
                # 全体に一様分布を割り当て
                label = np.full((height, width), 1.0 / (height * width), dtype=np.float32)
                
        else:
            x, y = coord
            if 0 <= x < width and 0 <= y < height:
                # ボール周辺のみにガウス分布を計算
                x_min = max(0, int(x - 3 * sigma))
                x_max = min(width, int(x + 3 * sigma) + 1)
                y_min = max(0, int(y - 3 * sigma))
                y_max = min(height, int(y + 3 * sigma) + 1)

                for i in range(y_min, y_max):
                    for j in range(x_min, x_max):
                        dist_squared = (x - j) ** 2 + (y - i) ** 2
                        label[i, j] = np.exp(-dist_squared / (2 * sigma ** 2))

                # 最大値で正規化
                label /= label.max()

        # ラベルを平坦化して追加
        labels.append(label.flatten())

    return np.array(labels, dtype=np.float32)

if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\kamim\Downloads\Dataset\Dataset\game1\Clip1\Label.csv")
    # ラベル生成（ヒートマップ形式）
    labels = create_labels_with_heatmap_optimized(df, output_shape=(360, 640), sigma=3)

    # サンプルデータ
    image_dir = r"C:\Users\kamim\Downloads\Dataset\Dataset\game1\Clip1"
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])

    # ヒートマップを (height, width) に戻す
    heatmaps = [label.reshape(360, 640) for label in labels]

    # 再生
    play_video_with_heatmap(image_paths, heatmaps, fps=30)
