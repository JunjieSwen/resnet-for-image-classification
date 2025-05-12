import os
import argparse
import random
import json
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw
from collections import Counter
from matplotlib import font_manager
# 自动选择第一个可用的中文字体
for font in font_manager.fontManager.ttflist:
    if any(key in font.name.lower() for key in ['noto', 'zenhei', 'droid']):
        plt.rcParams['font.family'] = font.name
        break

# 手动指定优先级（推荐）
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',     # Google Noto简体中文
    'WenQuanYi Zen Hei',    # 文泉驿正黑
    'Droid Sans Fallback'   # 安卓兼容字体
]
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据集检查、统计和平衡工具')
    parser.add_argument('--source_dir', type=str, default='/home/sunjj/classify/ncz_cls_6/NCZ/data/big_defact_0508', help='源数据集目录')
    parser.add_argument('--output_dir', type=str, default='/home/sunjj/classify/ncz_cls_6/NCZ/data/balanced/big_defact_0508', help='平衡后的数据集输出目录')
    parser.add_argument('--min_count', type=int, default=200, help='每个类别的最小图像数量')
    parser.add_argument('--max_count', type=int, default=2000, help='每个类别的最大图像数量')
    parser.add_argument('--aug_factor', type=int, default=2, help='图像增强倍数，如果类别数量不足min_count')
    parser.add_argument('--plots_dir', type=str, default='/home/sunjj/classify/ncz_cls_6/NCZ/plots/big_defact_0508', help='统计图表保存目录')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
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

def count_images_per_class(source_dir):
    """统计每个类别的图像数量"""
    class_counts = {}
    class_images = {}
    invalid_images = {}
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"找到 {len(classes)} 个类别: {classes}")
    
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        
        # 获取该类别的所有图像
        all_images = [f for f in os.listdir(class_dir) 
                    if os.path.isfile(os.path.join(class_dir, f)) and 
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        
        # 过滤无效图像
        valid_images = []
        invalid_img_list = []
        
        for img_file in tqdm(all_images, desc=f"验证 {class_name} 中的图像"):
            img_path = os.path.join(class_dir, img_file)
            if is_valid_image(img_path):
                valid_images.append(img_file)
            else:
                invalid_img_list.append(img_file)
        
        class_counts[class_name] = len(valid_images)
        class_images[class_name] = valid_images
        invalid_images[class_name] = invalid_img_list
        
        print(f"类别 {class_name}: {len(valid_images)} 有效图像, {len(invalid_img_list)} 无效图像")
    
    return class_counts, class_images, invalid_images

def plot_class_distribution(class_counts, output_dir, title="类别分布", filename="class_distribution.png"):
    """绘制类别分布直方图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 排序类别以便更好地可视化
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [x[0] for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    bars = plt.bar(class_names, counts)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f"{height}", ha='center', va='bottom')
    
    # 设置图形属性
    plt.title(title)
    plt.xlabel('类别')
    plt.ylabel('图像数量')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"分布图已保存至: {output_path}")
    
    return output_path

def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声"""
    # 将PIL图像转换为numpy数组
    img_array = np.array(image)
    
    # 生成高斯噪声
    noise = np.random.normal(mean, sigma, img_array.shape)
    
    # 添加噪声
    noisy_img_array = img_array + noise
    
    # 裁剪到有效范围
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    
    # 转换回PIL图像
    return Image.fromarray(noisy_img_array)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """添加椒盐噪声"""
    # 将PIL图像转换为numpy数组
    img_array = np.array(image)
    
    # 添加盐噪声（白点）
    salt_mask = np.random.random(img_array.shape[:2]) < salt_prob
    if len(img_array.shape) == 3:  # 彩色图像
        for i in range(img_array.shape[2]):
            img_array[salt_mask, i] = 255
    else:  # 灰度图像
        img_array[salt_mask] = 255
    
    # 添加椒噪声（黑点）
    pepper_mask = np.random.random(img_array.shape[:2]) < pepper_prob
    if len(img_array.shape) == 3:  # 彩色图像
        for i in range(img_array.shape[2]):
            img_array[pepper_mask, i] = 0
    else:  # 灰度图像
        img_array[pepper_mask] = 0
    
    # 转换回PIL图像
    return Image.fromarray(img_array)

def perspective_transform(image):
    """透视变换"""
    width, height = image.size
    
    # 定义四个顶点的偏移量 (小偏移以避免过度扭曲)
    offset = min(width, height) * 0.1
    max_offset = min(width, height) * 0.2
    
    # 定义变换前的四个点 (左上, 右上, 右下, 左下)
    coeffs = (
        0, 0,  # 左上
        width, 0,  # 右上
        width, height,  # 右下
        0, height  # 左下
    )
    
    # 定义变换后的四个点 (添加随机偏移)
    coeffs_mod = (
        0 + random.uniform(0, offset), 0 + random.uniform(0, offset),  # 左上
        width - random.uniform(0, offset), 0 + random.uniform(0, offset),  # 右上
        width - random.uniform(0, offset), height - random.uniform(0, offset),  # 右下
        0 + random.uniform(0, offset), height - random.uniform(0, offset)  # 左下
    )
    
    # 应用透视变换
    return image.transform(image.size, Image.PERSPECTIVE, coeffs_mod, Image.BICUBIC)

def elastic_transform(image):
    """弹性变换，模拟自然形变"""
    img_array = np.array(image)
    
    # 创建位移场
    shape = img_array.shape[:2]
    dx = np.random.rand(*shape) * 2 - 1
    dy = np.random.rand(*shape) * 2 - 1
    
    # 平滑位移场
    dx = dx * 5
    dy = dy * 5
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # 应用位移场
    if len(img_array.shape) == 3:  # 彩色图像
        distorted = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            distorted[:, :, i] = np.reshape(
                img_array[:, :, i][indices[0], indices[1]],
                shape
            )
    else:  # 灰度图像
        distorted = np.reshape(
            img_array[indices[0], indices[1]],
            shape
        )
    
    # 确保值在有效范围内
    distorted = np.clip(distorted, 0, 255).astype(np.uint8)
    
    return Image.fromarray(distorted)

def random_erase(image, erase_count=1, area_ratio=0.05):
    """随机擦除 (Random Erasing)"""
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    for _ in range(erase_count):
        # 确定擦除区域大小 (相对于原图的比例)
        erase_area = width * height * area_ratio
        aspect_ratio = random.uniform(0.3, 3)
        
        # 计算擦除区域的宽和高
        erase_width = int(np.sqrt(erase_area * aspect_ratio))
        erase_height = int(np.sqrt(erase_area / aspect_ratio))
        
        # 确保擦除区域不会超出图像边界
        erase_width = min(erase_width, width)
        erase_height = min(erase_height, height)
        
        # 随机选择擦除区域左上角位置
        x = random.randint(0, width - erase_width)
        y = random.randint(0, height - erase_height)
        
        # 用随机颜色或黑色/白色填充擦除区域
        if random.random() < 0.5:
            # 随机颜色
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            # 黑色或白色
            color = (0, 0, 0) if random.random() < 0.5 else (255, 255, 255)
        
        # 绘制矩形
        draw.rectangle([x, y, x + erase_width, y + erase_height], fill=color)
    
    return img

def apply_augmentation(image, aug_type):
    """对图像应用不同类型的增强"""
    if aug_type == 0:
        # 水平翻转
        return ImageOps.mirror(image)
    elif aug_type == 1:
        # 垂直翻转
        return ImageOps.flip(image)
    elif aug_type == 2:
        # 亮度增强
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)
    elif aug_type == 3:
        # 亮度降低
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(0.8)
    elif aug_type == 4:
        # 对比度增强
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)
    elif aug_type == 5:
        # 对比度降低
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(0.8)
    elif aug_type == 6:
        # 色彩增强
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.2)
    elif aug_type == 7:
        # 色彩降低
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(0.8)
    elif aug_type == 8:
        # 锐度增强
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)
    elif aug_type == 9:
        # 锐度降低
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(0.5)
    elif aug_type == 10:
        # 旋转90度
        return image.rotate(90, expand=True)
    elif aug_type == 11:
        # 旋转180度
        return image.rotate(180)
    elif aug_type == 12:
        # 旋转270度
        return image.rotate(270, expand=True)
    elif aug_type == 13:
        # 随机裁剪再放大
        width, height = image.size
        crop_size = min(width, height) * 0.85
        left = random.uniform(0, width - crop_size)
        top = random.uniform(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize(image.size, Image.LANCZOS)
    elif aug_type == 14:
        # 高斯模糊
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif aug_type == 15:
        # 边缘增强
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif aug_type == 16:
        # 轮廓滤镜
        return image.filter(ImageFilter.CONTOUR)
    elif aug_type == 17:
        # 浮雕效果
        return image.filter(ImageFilter.EMBOSS)
    elif aug_type == 18:
        # 添加高斯噪声
        return add_gaussian_noise(image)
    elif aug_type == 19:
        # 添加椒盐噪声
        return add_salt_pepper_noise(image)
    elif aug_type == 20:
        # 透视变换
        return perspective_transform(image)
    elif aug_type == 21:
        # 随机擦除
        return random_erase(image)
    elif aug_type == 22:
        # 小角度旋转 (±15度)
        angle = random.uniform(-15, 15)
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)
    elif aug_type == 23:
        # 自适应直方图均衡化 (PIL没有内置此功能，这里用模糊+锐化模拟)
        return image.filter(ImageFilter.GaussianBlur(radius=0.5)).filter(ImageFilter.SHARPEN)
    else:
        # 默认返回原图
        return image

def augment_class(class_name, source_dir, output_dir, source_images, target_count, aug_factor=5):
    """对类别进行图像增强，直到达到目标数量"""
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # 先复制所有原始图像
    for img_file in tqdm(source_images, desc=f"复制原始图像 {class_name}"):
        src_path = os.path.join(source_dir, class_name, img_file)
        dst_path = os.path.join(output_dir, class_name, img_file)
        shutil.copy2(src_path, dst_path)
    
    current_count = len(source_images)
    needed_count = target_count - current_count
    
    if needed_count <= 0:
        return current_count
    
    # 确定每个原始图像需要增强的次数
    aug_rounds = min(aug_factor, (needed_count // current_count) + 1)
    
    # 进行图像增强
    augmented_count = 0
    aug_types = [2,3,4,5,6,7,8,9,14,15,18,19,22]  # 24种不同的增强方式
    
    for img_file in tqdm(source_images, desc=f"增强图像 {class_name}"):
        src_path = os.path.join(source_dir, class_name, img_file)
        
        try:
            with Image.open(src_path) as original_img:
                original_img = original_img.convert('RGB')
                file_name, file_ext = os.path.splitext(img_file)
                
                # 对每个图像应用多轮增强
                for i in range(aug_rounds):
                    if augmented_count >= needed_count:
                        break
                    
                    # 随机选择增强类型
                    aug_type = random.choice(aug_types)
                    augmented_img = apply_augmentation(original_img, aug_type)
                    
                    # 保存增强后的图像
                    aug_filename = f"{file_name}_aug_{i}_{aug_type}{file_ext}"
                    aug_path = os.path.join(output_dir, class_name, aug_filename)
                    augmented_img.save(aug_path)
                    
                    augmented_count += 1
                    
                    # 如果已经达到目标数量，就停止增强
                    if current_count + augmented_count >= target_count:
                        break
        except Exception as e:
            print(f"增强图像时出错: {src_path}, 错误: {e}")
    
    return current_count + augmented_count

def sample_class(class_name, source_dir, output_dir, source_images, target_count):
    """对类别进行随机采样，使图像数量不超过目标数量"""
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # 随机采样
    sampled_images = random.sample(source_images, target_count)
    
    # 复制采样的图像
    for img_file in tqdm(sampled_images, desc=f"采样图像 {class_name}"):
        src_path = os.path.join(source_dir, class_name, img_file)
        dst_path = os.path.join(output_dir, class_name, img_file)
        shutil.copy2(src_path, dst_path)
    
    return len(sampled_images)

def balance_dataset(source_dir, output_dir, min_count, max_count, aug_factor, plots_dir, random_seed=42):
    """平衡数据集，使每个类别的图像数量在指定范围内"""
    random.seed(random_seed)
    
    # 统计每个类别的图像数量
    class_counts, class_images, invalid_images = count_images_per_class(source_dir)
    
    # 绘制原始分布图
    original_dist_plot = plot_class_distribution(
        class_counts, 
        plots_dir, 
        "原始数据集类别分布", 
        "original_distribution.png"
    )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 平衡后的类别数量
    balanced_counts = {}
    
    # 处理每个类别
    for class_name, count in class_counts.items():
        if count < min_count:
            # 如果类别数量少于最小值，进行增强
            print(f"对类别 {class_name} 进行增强 (当前: {count}, 目标: {min_count})")
            final_count = augment_class(
                class_name, 
                source_dir, 
                output_dir, 
                class_images[class_name], 
                min_count, 
                aug_factor
            )
        elif count > max_count:
            # 如果类别数量多于最大值，进行采样
            print(f"对类别 {class_name} 进行采样 (当前: {count}, 目标: {max_count})")
            final_count = sample_class(
                class_name, 
                source_dir, 
                output_dir, 
                class_images[class_name], 
                max_count
            )
        else:
            # 如果类别数量在范围内，直接复制
            print(f"类别 {class_name} 数量适中，直接复制 (当前: {count})")
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
            for img_file in tqdm(class_images[class_name], desc=f"复制图像 {class_name}"):
                src_path = os.path.join(source_dir, class_name, img_file)
                dst_path = os.path.join(output_dir, class_name, img_file)
                shutil.copy2(src_path, dst_path)
            final_count = count
        
        balanced_counts[class_name] = final_count
    
    # 绘制平衡后的分布图
    balanced_dist_plot = plot_class_distribution(
        balanced_counts, 
        plots_dir, 
        "平衡后数据集类别分布", 
        "balanced_distribution.png"
    )
    
    # 保存统计信息
    stats = {
        "original_counts": class_counts,
        "balanced_counts": balanced_counts,
        "invalid_counts": {k: len(v) for k, v in invalid_images.items()},
        "min_count": min_count,
        "max_count": max_count,
        "aug_factor": aug_factor
    }
    
    stats_path = os.path.join(plots_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    
    print(f"数据集统计信息已保存至: {stats_path}")
    
    # 绘制对比图
    plot_comparison(class_counts, balanced_counts, plots_dir)
    
    return stats

def plot_comparison(original_counts, balanced_counts, output_dir):
    """绘制平衡前后的对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有类别
    all_classes = sorted(set(list(original_counts.keys()) + list(balanced_counts.keys())))
    
    # 准备数据
    original_values = [original_counts.get(cls, 0) for cls in all_classes]
    balanced_values = [balanced_counts.get(cls, 0) for cls in all_classes]
    
    # 设置图形尺寸
    plt.figure(figsize=(15, 10))
    
    # 设置柱状图
    x = np.arange(len(all_classes))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, original_values, width, label='原始数量')
    plt.bar(x + width/2, balanced_values, width, label='平衡后数量')
    
    # 设置图形属性
    plt.xlabel('类别')
    plt.ylabel('图像数量')
    plt.title('数据集平衡前后对比')
    plt.xticks(x, all_classes, rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加水平线表示最小和最大阈值
    min_counts = min(balanced_values)
    max_counts = max(balanced_values)
    plt.axhline(y=min_counts, color='r', linestyle='--', alpha=0.5, label=f'最小阈值 ({min_counts})')
    plt.axhline(y=max_counts, color='g', linestyle='--', alpha=0.5, label=f'最大阈值 ({max_counts})')
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"对比图已保存至: {output_path}")

def main():
    """主函数"""
    args = parse_args()
    
    print(f"开始对数据集进行检查和平衡...")
    print(f"源数据集目录: {args.source_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"类别图像数量范围: {args.min_count} - {args.max_count}")
    print(f"图像增强倍数: {args.aug_factor}")
    
    # 平衡数据集
    stats = balance_dataset(
        args.source_dir,
        args.output_dir,
        args.min_count,
        args.max_count,
        args.aug_factor,
        args.plots_dir,
        args.random_seed
    )
    
    # 输出总结
    original_total = sum(stats["original_counts"].values())
    balanced_total = sum(stats["balanced_counts"].values())
    
    print("\n数据集平衡完成!")
    print(f"原始数据集: {len(stats['original_counts'])} 个类别, {original_total} 张图像")
    print(f"平衡后数据集: {len(stats['balanced_counts'])} 个类别, {balanced_total} 张图像")
    print(f"使用了24种不同的数据增强方式")
    
    if balanced_total > original_total:
        print(f"通过图像增强增加了 {balanced_total - original_total} 张图像")
    else:
        print(f"通过随机采样减少了 {original_total - balanced_total} 张图像")
    
    print(f"统计图表和数据已保存到: {args.plots_dir}")
    print(f"平衡后的数据集已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 