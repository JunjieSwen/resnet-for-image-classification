import os
import shutil
import argparse
import random
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='按自定义比例随机划分数据集')
    parser.add_argument('--source_dir', type=str, required=True, help='源数据集目录')
    parser.add_argument('--output_dir', type=str, default='../data/split_data', help='输出目录')
    parser.add_argument('--splits', type=str, required=True, help='划分比例，格式为"名称:比例,名称:比例"，例如"train:0.7,val:0.2,test:0.1"')
    parser.add_argument('--img_size', type=int, default=None, help='调整图像大小（可选）')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--copy_mode', choices=['copy', 'move', 'symlink'], default='copy', 
                        help='文件处理模式：copy(复制)、move(移动)或symlink(创建符号链接)')
    return parser.parse_args()

def is_valid_image(image_path):
    """检查图像是否有效"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"无效图像: {image_path}, 错误: {e}")
        return False

def parse_splits(splits_str):
    """解析划分比例字符串"""
    splits = {}
    total_ratio = 0
    
    # 解析划分比例
    for split_pair in splits_str.split(','):
        name, ratio = split_pair.split(':')
        name = name.strip()
        ratio = float(ratio.strip())
        splits[name] = ratio
        total_ratio += ratio
    
    # 验证比例总和是否接近1
    if not (0.99 <= total_ratio <= 1.01):
        print(f"警告: 划分比例总和为 {total_ratio}，与 1.0 有偏差")
        # 归一化比例
        for name in splits:
            splits[name] = splits[name] / total_ratio
        print(f"已归一化比例: {splits}")
    
    return splits

def create_directory_structure(output_dir, split_names):
    """创建输出目录结构"""
    # 创建主目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建各个划分目录
    split_dirs = {}
    for name in split_names:
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        split_dirs[name] = split_dir
    
    return split_dirs

def split_dataset(source_dir, split_dirs, splits, img_size, copy_mode, random_seed=42):
    """按比例划分数据集"""
    # 设置随机种子
    random.seed(random_seed)
    
    # 获取源目录中的所有类别
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"找到 {len(classes)} 个类别: {classes}")
    
    # 创建类别目录
    for class_name in classes:
        for split_dir in split_dirs.values():
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    # 处理每个类别
    all_splits_count = {name: 0 for name in splits.keys()}
    class_stats = {}
    
    for class_name in classes:
        print(f"处理类别: {class_name}")
        class_dir = os.path.join(source_dir, class_name)
        
        # 获取该类别的所有图像
        image_files = [f for f in os.listdir(class_dir) 
                      if os.path.isfile(os.path.join(class_dir, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        
        # 过滤无效图像
        valid_images = []
        for img_file in tqdm(image_files, desc=f"验证 {class_name} 中的图像"):
            img_path = os.path.join(class_dir, img_file)
            if is_valid_image(img_path):
                valid_images.append(img_file)
            else:
                print(f"跳过无效图像: {img_path}")
        
        print(f"类别 {class_name} 中有效图像数量: {len(valid_images)}/{len(image_files)}")
        
        # 随机打乱图像顺序
        random.shuffle(valid_images)
        
        # 计算划分点
        total_images = len(valid_images)
        split_points = [0]
        cumulative_ratio = 0
        
        for name, ratio in splits.items():
            cumulative_ratio += ratio
            split_point = int(total_images * cumulative_ratio)
            split_points.append(split_point)
        
        # 划分数据集并记录每个划分的图像数量
        split_images = {}
        class_split_stats = {}
        
        for i, name in enumerate(splits.keys()):
            start = split_points[i]
            end = split_points[i+1]
            split_images[name] = valid_images[start:end]
            class_split_stats[name] = len(split_images[name])
            all_splits_count[name] += len(split_images[name])
        
        # 记录类别统计信息
        class_stats[class_name] = class_split_stats
        
        # 输出划分统计
        split_info = ", ".join([f"{name}: {len(imgs)}张" for name, imgs in split_images.items()])
        print(f"类别 {class_name} 划分: {split_info}")
        
        # 处理图像
        for split_name, images in split_images.items():
            target_dir = split_dirs[split_name]
            
            for img_file in tqdm(images, desc=f"处理 {split_name}/{class_name} 中的图像"):
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(target_dir, class_name, img_file)
                
                # 根据不同的复制模式处理文件
                if copy_mode == 'copy':
                    if img_size:
                        # 如果需要调整图像大小
                        with Image.open(src_path) as img:
                            img.thumbnail((img_size, img_size))
                            img.save(dst_path)
                    else:
                        # 直接复制
                        shutil.copy2(src_path, dst_path)
                        
                elif copy_mode == 'move':
                    if img_size:
                        # 如果需要调整图像大小
                        with Image.open(src_path) as img:
                            img.thumbnail((img_size, img_size))
                            img.save(dst_path)
                        os.remove(src_path)
                    else:
                        # 直接移动
                        shutil.move(src_path, dst_path)
                        
                elif copy_mode == 'symlink':
                    # 创建符号链接 (不支持调整大小)
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    
                    # 获取相对路径
                    src_relative = os.path.relpath(src_path, os.path.dirname(dst_path))
                    try:
                        os.symlink(src_relative, dst_path)
                    except:
                        # Windows可能需要管理员权限，如果失败则改为复制
                        print(f"创建符号链接失败，改为复制文件: {dst_path}")
                        shutil.copy2(src_path, dst_path)
    
    return all_splits_count, class_stats

def save_stats(output_dir, all_splits_count, class_stats):
    """保存数据集统计信息"""
    stats = {
        "summary": all_splits_count,
        "classes": class_stats
    }
    
    stats_path = os.path.join(output_dir, 'split_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    
    print(f"\n数据集统计信息已保存至: {stats_path}")

def main():
    """主函数"""
    args = parse_args()
    
    # 解析划分比例
    splits = parse_splits(args.splits)
    print(f"数据集将被划分为: {splits}")
    
    # 创建输出目录结构
    split_dirs = create_directory_structure(args.output_dir, splits.keys())
    
    # 划分数据集
    all_splits_count, class_stats = split_dataset(
        args.source_dir, 
        split_dirs, 
        splits, 
        args.img_size,
        args.copy_mode,
        args.random_seed
    )
    
    # 保存统计信息
    save_stats(args.output_dir, all_splits_count, class_stats)
    
    # 总结划分结果
    print("\n数据集划分完成!")
    total_images = sum(all_splits_count.values())
    
    for name, count in all_splits_count.items():
        actual_ratio = count / total_images if total_images > 0 else 0
        expected_ratio = splits[name]
        print(f"{name}: {count} 张图像 (实际比例: {actual_ratio:.4f}, 预期比例: {expected_ratio:.4f})")
    
    print(f"总计: {total_images} 张图像")
    print(f"数据已{args.copy_mode}到 {args.output_dir} 目录")

if __name__ == "__main__":
    main() 