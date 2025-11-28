import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


def binary_metrics(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    eps = 1e-7

    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)

    return acc, precision, recall, iou, dice


def predict_mask(model, img_path):
    model.eval()
    img = Image.open(img_path).convert("L")

    transform = T.Compose([
        # T.Resize((256, 256)),
        T.ToTensor()
    ])

    x = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).cpu().numpy()[0, 0]

    return pred


def evaluate(model_path, img_dir, gt_dir):

    print("Loading model:", model_path)

    model = torch.load(model_path)
    model = model.cuda()
    model.eval()

    imgs = sorted(os.listdir(img_dir))
    gts = sorted(os.listdir(gt_dir))

    results = []

    for im, gt in zip(imgs, gts):
        img_path = os.path.join(img_dir, im)
        gt_path = os.path.join(gt_dir, gt)

        pred = predict_mask(model, img_path)

        gt_mask = np.array(Image.open(gt_path))
        if gt_mask.max() > 1:
            gt_mask = gt_mask / 255
        gt_mask = (gt_mask > 0.5).astype(np.uint8)

        metrics = binary_metrics(pred, gt_mask)
        results.append(metrics)

        print(f"{im}: IoU={metrics[3]:.4f}, Dice={metrics[4]:.4f}")

    results = np.array(results)
    print("\n====== Final Average Metrics ======")
    print(f"Accuracy:  {results[:,0].mean():.4f}")
    print(f"Precision: {results[:,1].mean():.4f}")
    print(f"Recall:    {results[:,2].mean():.4f}")
    print(f"IoU:       {results[:,3].mean():.4f}")
    print(f"Dice:      {results[:,4].mean():.4f}")


if __name__ == "__main__":
    evaluate(
        model_path="Unet-epochs200.pth",
        img_dir="data/val/img",
        gt_dir="data/val/gt"
    )
