import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import shutil


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用ResNet50模型预测图像')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--image_path', type=str, required=True, help='要预测的图像文件夹路径')
    parser.add_argument('--class_names_path', type=str, required=True, help='类别名称JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='输出目录')
    return parser.parse_args()


def load_model(model_path, num_classes):
    """加载模型"""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))

    # 设置为评估模式
    model.eval()

    return model


def load_class_names(class_names_path):
    """加载类别名称"""
    if class_names_path and os.path.isfile(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        return class_names
    else:
        return None


def preprocess_image(image_path):
    """预处理图像"""
    # 数据转换
    preprocess = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 打开图像
    image = Image.open(image_path).convert('RGB')

    # 应用预处理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度

    return input_batch, image


def predict_image(model, image_path, class_names, device, topk=3):
    """预测单张图像"""
    # 预处理图像
    input_batch, original_image = preprocess_image(image_path)
    input_batch = input_batch.to(device)

    # 预测
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取前k个预测
    topk_probs, topk_indices = torch.topk(probabilities, topk)

    # 转换为numpy数组
    topk_probs = topk_probs.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()

    # 获取类别名称
    if class_names:
        topk_class_names = [class_names[idx] for idx in topk_indices]
    else:
        topk_class_names = [f"类别 {idx}" for idx in topk_indices]

    return topk_probs, topk_class_names, original_image


def visualize_prediction(image, probs, classes, output_path=None):
    """可视化预测结果"""
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示图像
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('输入图像')

    # 显示预测结果
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probs * 100, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.set_xlabel('概率 (%)')
    ax2.set_title('预测结果')
    ax2.set_xlim(0, 100)

    # 调整布局
    plt.tight_layout()

    # 保存图形
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"预测结果已保存至: {output_path}")

    plt.close()


def process_directory(model, directory_path, class_names, device, output_dir):
    """处理目录中的所有图像并保存到对应的类别文件夹"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # 获取目录中的所有图像
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                   if os.path.isfile(os.path.join(directory_path, f)) and
                   f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"警告: 在目录 {directory_path} 中未找到图像文件")
        return

    # 处理每张图像
    for image_path in image_files:
        # 预测图像
        input_batch, _ = preprocess_image(image_path)
        input_batch = input_batch.to(device)

        # 预测
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 获取预测结果
        top_prob, top_idx = torch.max(probabilities, 0)
        top_prob = top_prob.cpu().numpy()
        top_idx = top_idx.cpu().numpy()

        # 获取类别名称
        predicted_class = class_names[top_idx] if class_names else f"类别 {top_idx}"

        # 创建类别文件夹
        class_dir = os.path.join(output_dir, predicted_class)
        os.makedirs(class_dir, exist_ok=True)

        # 复制图片到对应类别文件夹
        dest_path = os.path.join(class_dir, os.path.basename(image_path))
        shutil.copy2(image_path, dest_path)

        print(f"图像 {os.path.basename(image_path)} 预测为 {predicted_class} (概率: {top_prob * 100:.2f}%)")


def main():
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 加载类别名称
    class_names = load_class_names(args.class_names_path)
    if not class_names:
        print("错误: 无法加载类别名称文件")
        return

    # 确定类别数量
    num_classes = len(class_names)

    # 检查CUDA是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(args.model_path, num_classes)
    model = model.to(device)

    # 处理图像目录
    if os.path.isdir(args.image_path):
        process_directory(model, args.image_path, class_names, device, args.output_dir)
    else:
        print(f"错误: 输入路径不是目录: {args.image_path}")


if __name__ == "__main__":
    main() 