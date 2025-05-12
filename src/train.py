import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import true
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timm

from matplotlib import font_manager

for font in font_manager.fontManager.ttflist:
    if any(key in font.name.lower() for key in ['noto', 'zenhei', 'droid']):
        plt.rcParams['font.family'] = font.name
        break

# 手动指定优先级（推荐）
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',  # Google Noto简体中文
    'WenQuanYi Zen Hei',  # 文泉驿正黑
    'Droid Sans Fallback'  # 安卓兼容字体
]
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
from tqdm import tqdm


# from pytorchtools import EarlyStopping

# from model import CBAM


def parse_args():
    parser = argparse.ArgumentParser(description='使用ResNet50进行图像分类')
    parser.add_argument('--data_dir', type=str, default='../data/split_data/big_defact_0508_enhance', help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器的工作线程数')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoint/models_efficientnet', help='模型保存路径')
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--patience', type=int, default=10, help='patience')

    # device
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用gpu进行训练')
    parser.add_argument('--gpu', type=int, required=True, help='gpu')
    parser.add_argument('--use_multi_gpu', type=int, default=True, help='use multi gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multi gpus')

    return parser.parse_args()


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存最佳模型"""
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型 ...')
            self.val_loss_min = val_loss
            return True
        return False


def get_data_transforms():
    """定义数据转换和增强"""
    # 训练集的数据增强
    train_transforms = transforms.Compose([
        transforms.Resize((448, 448)),
        # transforms.RandomResizedCrop((448,448)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 验证集和测试集的数据转换
    val_test_transforms = transforms.Compose([
        transforms.Resize((448, 448)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms


def load_data(data_dir, batch_size, num_workers):
    """加载数据集"""
    train_transforms, val_test_transforms = get_data_transforms()

    # 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transforms)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 获取类别名称
    class_names = train_dataset.classes

    print(f"数据集加载完成，类别数: {len(class_names)}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别信息: {class_names}")

    return train_loader, val_loader, test_loader, class_names


# def build_model(num_classes, pretrained=True, dropout_prob=0.3):  # 调整默认Dropout概率
#     model = models.resnet50(pretrained=pretrained)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model
def build_model(num_classes, pretrained=True, dropout_prob=0.3):  # 调整默认Dropout概率
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def train_model(args, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device,
                model_save_dir):
    """训练模型"""
    best_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=0)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 使用tqdm创建进度条
        train_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 更新进度条
            train_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / inputs.size(0))

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # 使用tqdm创建进度条
        val_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{num_epochs}")

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新进度条
                val_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / inputs.size(0))

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())

        print(f'验证 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 如果使用ReduceLROnPlateau，在这里调用scheduler.step()并传入验证损失
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)

        # 早停检查
        early_stopping(epoch_loss, model)
        
        # 保存最佳模型
        if early_stopping.save_checkpoint(epoch_loss, model):
            # 如果目录不存在，创建它
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'最佳模型已保存: {model_path}')

        if early_stopping.early_stop:
            print("触发早停！")
            # 恢复最佳模型
            model.load_state_dict(early_stopping.best_model_state)
            break

        print()

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'最终模型已保存: {final_model_path}')

    # 返回训练历史
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }

    return history


def plot_training_history(history, save_dir):
    """绘制训练历史图表"""
    # 创建保存图表的目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_path)
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()

    print(f"训练历史图表已保存到: {save_dir}")


def main():
    args = parse_args()

    # 定义训练设备
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    print(f"使用设备: {args.device}")
    # 多卡训练
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        devices_id = args.devices.split(',')
        args.devices_ids = [int(id_) for id_ in devices_id]
        args.gpu = args.devices_ids[0]

    # 加载数据
    train_loader, val_loader, test_loader, class_names = load_data(args.data_dir, args.batch_size, args.num_workers)

    # 构建模型
    model = build_model(num_classes=len(class_names), pretrained=args.pretrained)
    model = model.to(args.device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 学习率调度器（动态调整）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # 训练模型
    start_time = time.time()
    history = train_model(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        model_save_dir=args.model_save_dir
    )
    end_time = time.time()

    print(f"训练完成，总耗时: {(end_time - start_time) / 60:.2f} 分钟")

    # 绘制训练历史
    plot_training_history(history, args.model_save_dir)

    # 加载最佳模型并在测试集上评估
    best_model_path = os.path.join(args.model_save_dir, f'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))

    # 在测试集上评估模型
    from evaluate import evaluate_model
    test_acc, test_report = evaluate_model(model, test_loader, args.device)

    print(f"测试集准确率: {test_acc:.4f}")
    print("分类报告:")
    print(test_report)
    # 释放内存
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 