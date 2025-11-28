import os
from PIL import Image
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

def process_images(directory):
    if not os.path.isdir(directory):
        print(f"目录不存在: {directory}")
        return

    for filename in os.listdir(directory):
        if filename.lower().endswith(IMG_EXTS):
            filepath = os.path.join(directory, filename)

            try:
                img = Image.open(filepath)
                arr = np.array(img)

                # 判断数组中是否有大于 0 的像素
                if np.any(arr > 0):
                    print("==== 存在大于 0 的图片 ====")
                    print("文件名:", filename)
                    print("数组形状:", arr.shape)

                    # 打印数组（可选：完整打印）
                    np.set_printoptions(threshold=None, linewidth=1000000)
                    # ★ 完整无折叠打印数组 ★
                    arr_str = np.array2string(
                        arr,
                        max_line_width=10 ** 9,
                        threshold=arr.size,
                        edgeitems=arr.size
                    )

                    print(arr_str)
                    print()

            except Exception as e:
                print(f"无法读取 {filepath}: {e}")

def main():
    target_dir = "/home/chxing/work/ecnu/computer_version/example-pytorch/data/val/gt/"
    process_images(target_dir)

if __name__ == "__main__":
    main()
