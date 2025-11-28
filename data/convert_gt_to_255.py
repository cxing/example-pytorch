import os
from PIL import Image
import numpy as np

def convert_gt_to_255(gt_dir, save_dir=None):
    """
    将 GT 文件夹中的 0/1 mask 全部转换为 0/255
    gt_dir: 原 GT 文件夹，如 'data/train/gt'
    save_dir: 保存目录；为 None 则原地覆盖
    """

    if save_dir is None:
        save_dir = gt_dir  # 原地覆盖
    else:
        os.makedirs(save_dir, exist_ok=True)

    for fname in sorted(os.listdir(gt_dir)):
        fpath = os.path.join(gt_dir, fname)

        # 读取 mask
        mask = Image.open(fpath).convert("L")
        mask_np = np.array(mask)

        # 将 0/1 转成 0/255
        mask_np = (mask_np > 0).astype(np.uint8) * 255

        # 保存
        save_path = os.path.join(save_dir, fname)
        Image.fromarray(mask_np).save(save_path)

        print(f"Converted: {fname}")

def convert_gt_255_to_1(gt_dir, save_dir=None):
    """
    将 GT 掩膜从 0/255 转成 0/1
    gt_dir: 原 GT 文件夹
    save_dir: 转换后保存的目录；None=原地覆盖
    """

    if save_dir is None:
        save_dir = gt_dir
    else:
        os.makedirs(save_dir, exist_ok=True)

    for fname in sorted(os.listdir(gt_dir)):
        fpath = os.path.join(gt_dir, fname)

        # 读取
        mask = Image.open(fpath).convert("L")
        mask_np = np.array(mask)

        # ⭐ 核心转换：0/255 → 0/1
        mask_np = (mask_np > 127).astype(np.uint8)

        # 保存
        save_path = os.path.join(save_dir, fname)
        Image.fromarray(mask_np).save(save_path)

        print(f"Converted: {fname}")

if __name__ == "__main__":
    # 你想转换哪个目录就填哪个
    convert_gt_255_to_1("/home/chxing/work/ecnu/computer_version/example-pytorch/data/train/gt")
    convert_gt_255_to_1("/home/chxing/work/ecnu/computer_version/example-pytorch/data/val/gt")