import torch
import numpy as np
from PIL import Image
from model.net import UNet
import torchvision.transforms as T

def predict_mask_simple(model, img_path):
    img = Image.open(img_path).convert("L")

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    x = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).cpu().numpy()[0, 0]

    return pred


if __name__ == "__main__":
    model_path = "Unet-epochs200.pth"
    test_img = "data/val/img/Patient_02_019.png"   # 任意一个验证图像

    print("Loading model...")
    model = torch.load(model_path).cuda().eval()

    print("Predicting...")
    pred = predict_mask_simple(model, test_img)

    # ⭐ 最重要的判定（是否全黑）
    unique_vals = np.unique(pred)
    print("Unique values in prediction:", unique_vals)

    if len(unique_vals) == 1 and unique_vals[0] == 0:
        print("\n🔥 RESULT: 模型预测为【全黑掩膜】（没有任何前景）")
    else:
        print("\n🎉 RESULT: 模型【预测出了前景】！")

    # 可选：显示前 20 个数值
    print("\nPrediction sample values:", pred.flatten()[:20])
