import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    """在给定数据加载器上评估模型性能"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 使用tqdm显示进度
    eval_bar = tqdm(data_loader, desc="评估模型")
    
    with torch.no_grad():
        for inputs, labels in eval_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 生成分类报告
    target_names = data_loader.dataset.classes if hasattr(data_loader.dataset, 'classes') else None
    report = classification_report(all_labels, all_preds, target_names=target_names)
    
    return accuracy, report

def plot_confusion_matrix(model, data_loader, device, save_path=None):
    """绘制混淆矩阵"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 获取类别名称
    class_names = data_loader.dataset.classes if hasattr(data_loader.dataset, 'classes') else None
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.close()
    
    return cm

def visualize_predictions(model, data_loader, device, num_images=5, save_dir=None):
    """可视化模型预测结果"""
    # 如果保存目录不存在，创建它
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 获取类别名称
    class_names = data_loader.dataset.classes if hasattr(data_loader.dataset, 'classes') else None
    
    # 收集所有数据
    all_images = []
    all_labels = []
    
    # 收集数据
    with torch.no_grad():
        for images, labels in data_loader:
            all_images.append(images)
            all_labels.append(labels)
    
    # 合并所有批次的数据
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 随机选择样本
    indices = torch.randperm(len(all_images))[:num_images]
    images = all_images[indices].to(device)
    labels = all_labels[indices].to(device)
    
    # 获取模型预测
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    images = images * std + mean
    images = images.clamp(0, 1)
    
    # 绘制图像和预测
    fig = plt.figure(figsize=(15, 5 * (num_images // 3 + 1)))
    
    for i in range(num_images):
        ax = fig.add_subplot(num_images // 3 + 1, 3, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        title_color = 'green' if preds[i] == labels[i] else 'red'
        
        if class_names:
            title = f'预测: {class_names[preds[i]]}\n真实: {class_names[labels[i]]}'
        else:
            title = f'预测: {preds[i].item()}\n真实: {labels[i].item()}'
        
        ax.set_title(title, color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        save_path = os.path.join(save_dir, 'prediction_visualization.png')
        plt.savefig(save_path)
        print(f"预测可视化已保存至: {save_path}")
    
    plt.close()

def main():
    import argparse
    import torch
    from torchvision import models
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import os
    
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, required=True, help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器的工作线程数')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查CUDA是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据变换
    test_transforms = transforms.Compose([
        transforms.Resize((448,448)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    test_dataset = datasets.ImageFolder(args.data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 获取类别数量
    num_classes = len(test_dataset.classes)
    
    # 加载模型
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # 评估模型
    accuracy, report = evaluate_model(model, test_loader, device)
    print(f"测试集准确率: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    
    # 保存分类报告
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"测试集准确率: {accuracy:.4f}\n")
        f.write("分类报告:\n")
        f.write(report)
    print(f"分类报告已保存至: {report_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(model, test_loader, device, save_path=cm_path)
    
    # 可视化预测
    visualize_predictions(model, test_loader, device, num_images=10, save_dir=args.output_dir)

if __name__ == "__main__":
    main() 