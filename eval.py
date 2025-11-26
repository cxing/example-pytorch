# eval.py
import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model.net import UNet


# -----------------------
# 1. 验证集 Dataset（不做随机增强）
# -----------------------
class ValDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".png")
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = self.to_tensor(img)                # [0,1]
        mask = self.to_tensor(mask)              # [0,1]
        mask = (mask > 0.5).float()              # 二值化到 {0,1}

        return img, mask


# -----------------------
# 2. 各种评价指标
# -----------------------
def dice_coeff(pred, target, eps=1e-6):
    """
    pred, target: [B,1,H,W] with values 0 or 1
    """
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def pixel_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()


def precision_recall(pred, target, eps=1e-6):
    tp = ((pred == 1) & (target == 1)).float().sum()
    fp = ((pred == 1) & (target == 0)).float().sum()
    fn = ((pred == 0) & (target == 1)).float().sum()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision.item(), recall.item()


# -----------------------
# 3. 验证主函数
# -----------------------
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    val_dataset = ValDataset(
        img_dir=os.path.join(args.data_root, "val", "img"),
        mask_dir=os.path.join(args.data_root, "val", "gt"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # 如果保存的是 model 而不是 state_dict，直接用：
    #   model = torch.load(args.checkpoint, map_location=device)
    # 下面这行则可以改为：pass
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model = ckpt

    model.eval()

    dice_list = []
    iou_list = []
    acc_list = []
    prec_list = []
    rec_list = []

    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device)
            mask = mask.to(device)

            logits = model(img)                  # [B,1,H,W]
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            dice_list.append(dice_coeff(pred, mask))
            iou_list.append(iou_score(pred, mask))
            acc_list.append(pixel_accuracy(pred, mask))
            p, r = precision_recall(pred, mask)
            prec_list.append(p)
            rec_list.append(r)

    import numpy as np
    dice_mean = np.mean(dice_list)
    iou_mean = np.mean(iou_list)
    acc_mean = np.mean(acc_list)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)

    print("Validation results:")
    print(f"  Pixel Accuracy: {acc_mean:.4f}")
    print(f"  Dice Coefficient: {dice_mean:.4f}")
    print(f"  IoU: {iou_mean:.4f}")
    print(f"  Precision: {prec_mean:.4f}")
    print(f"  Recall: {rec_mean:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data",
                        help="root path of data folder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to trained model checkpoint, e.g. Unet-epochs200.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    evaluate(args)
