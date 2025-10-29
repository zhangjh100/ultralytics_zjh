import os
import cv2
from pathlib import Path
from ultralytics.utils.patches import imread  # 使用代码库中的图像读取函数，支持多语言文件名


def convert_single_to_3channel(input_dir, output_dir):
    """
    将指定文件夹下的所有单通道图像转换为3通道图像并保存

    Args:
        input_dir (str): 输入图像文件夹路径
        output_dir (str): 输出图像文件夹路径
    """
    # 创建输出文件夹
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件是否为图像
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)

            # 使用代码库中的imread函数读取图像
            img = imread(input_path)
            if img is None:
                print(f"无法读取图像: {input_path}")
                continue

            # 检查通道数，若为单通道则转换为3通道
            if img.ndim == 2:  # 单通道灰度图
                # 转换为3通道（BGR格式，OpenCV默认）
                img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                print(f"已将单通道图像转换为3通道: {filename}")
            elif img.shape[2] == 1:  # 形状为(H, W, 1)的单通道图
                img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                print(f"已将单通道图像转换为3通道: {filename}")
            else:  # 已经是3通道或更多通道
                if img.shape[2] > 3:
                    img_3ch = img[:, :, :3]  # 只取前3通道
                    print(f"已将多通道图像转为3通道: {filename}")
                else:
                    img_3ch = img
                    print(f"图像已是3通道: {filename}")

            # 保存转换后的图像
            output_path = os.path.join(output_dir, filename)
            # 使用OpenCV保存图像，处理中文路径
            cv2.imencode(os.path.splitext(filename)[1], img_3ch)[1].tofile(output_path)


# 使用示例
if __name__ == "__main__":
    input_directory = "/mnt/data1/zhangjh/ultralytics_zjh/datasets/acdc/val/labels_1"  # 替换为你的输入文件夹路径
    output_directory = "/mnt/data1/zhangjh/ultralytics_zjh/datasets/acdc/val/labels_3"  # 替换为你的输出文件夹路径

    convert_single_to_3channel(input_directory, output_directory)
    print("图像处理完成！")