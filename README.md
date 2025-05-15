# ResNet50 图像分类项目

这是一个使用ResNet50模型对自定义图像数据集进行分类的完整项目。项目包含了数据处理、模型训练、评估和预测的全过程。

## 项目结构

```
├── data/               # 数据集目录
│   ├── train/          # 训练集
│   ├── val/            # 验证集
│   └── test/           # 测试集
├── models/             # 模型保存目录
├── src/                # 源代码
│   ├── prepare_data.py     # 数据预处理脚本
│   ├── split_dataset.py    # 自定义比例数据集划分脚本
│   ├── balance_dataset.py  # 数据集平衡工具（检查、统计、增强/采样）
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── predict.py          # 预测脚本
│   └── extract_class_names.py  # 类别名称提取工具
└── requirements.txt    # 项目依赖
```

## 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

### 数据集目录结构

您需要准备一个包含多个类别的图像数据集，每个类别的图像放在单独的文件夹中。例如：

```
dataset/
├── 类别1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 类别2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

### 数据集划分

#### 方法一：使用固定比例划分（训练集、验证集、测试集）

使用`prepare_data.py`脚本将数据集划分为训练集、验证集和测试集：

```bash
python src/prepare_data.py --source_dir /path/to/your/dataset --output_dir ./data --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

参数说明：
- `--source_dir`: 源数据集目录
- `--output_dir`: 输出目录（默认为`./data`）
- `--train_ratio`: 训练集比例（默认为0.7）
- `--val_ratio`: 验证集比例（默认为0.15）
- `--test_ratio`: 测试集比例（默认为0.15）
- `--img_size`: 可选参数，调整图像大小
- `--random_seed`: 随机种子（默认为42）

#### 方法二：使用自定义比例划分

使用`split_dataset.py`脚本可以按照任意自定义比例划分数据集：

```bash
python src/split_dataset.py --source_dir /path/to/your/dataset --output_dir ./custom_data --splits "train:0.6,val:0.2,test:0.2"
```

参数说明：
- `--source_dir`: 源数据集目录
- `--output_dir`: 输出目录（默认为`./split_data`）
- `--splits`: 划分比例，格式为"名称:比例,名称:比例"，例如"train:0.7,val:0.2,test:0.1"
- `--img_size`: 可选参数，调整图像大小
- `--random_seed`: 随机种子（默认为42）
- `--copy_mode`: 文件处理模式，可选`copy`（复制，默认）、`move`（移动）或`symlink`（创建符号链接）

这个脚本更为灵活，允许您：
- 自定义任意数量的划分及其名称（不仅限于train/val/test）
- 选择文件处理方式（复制/移动/符号链接）
- 自动生成划分统计报告（保存为JSON文件）

例如，您可以创建更多的划分：

```bash
python src/split_dataset.py --source_dir /path/to/dataset --splits "train:0.6,val:0.15,test:0.15,extra:0.1"
```

或者使用符号链接以节省磁盘空间：

```bash
python src/split_dataset.py --source_dir /path/to/dataset --splits "train:0.8,test:0.2" --copy_mode symlink
```

### 数据集平衡工具

使用`balance_dataset.py`脚本可以检查数据集、统计类别分布、生成可视化图表，并平衡各个类别的图像数量：

```bash
python src/balance_dataset.py --source_dir /path/to/your/dataset --output_dir ./balanced_data --min_count 200 --max_count 2000
```

参数说明：
- `--source_dir`: 源数据集目录
- `--output_dir`: 平衡后的数据集输出目录（默认为`./balanced_data`）
- `--min_count`: 每个类别的最小图像数量（默认为200）
- `--max_count`: 每个类别的最大图像数量（默认为2000）
- `--aug_factor`: 图像增强倍数，如果类别数量不足min_count（默认为5）
- `--plots_dir`: 统计图表保存目录（默认为`./plots`）
- `--random_seed`: 随机种子（默认为42）

该工具的工作流程：

1. **检查和统计**：分析数据集中每个类别的图像数量，过滤无效图像。
2. **可视化**：生成数据集分布直方图，展示各类别图像数量。
3. **平衡处理**：
   - 对于图像数量少于`min_count`的类别，通过图像增强技术（如翻转、旋转、调整亮度/对比度等）扩充到至少`min_count`张。
   - 对于图像数量多于`max_count`的类别，随机采样`max_count`张图像。
   - 对于数量在范围内的类别，直接复制原图像。
4. **结果分析**：生成平衡前后对比图表，保存详细统计信息。

输出内容：
- 平衡后的数据集（保存在`output_dir`）
- 统计图表（类别分布直方图、平衡前后对比图）
- 数据集统计信息（JSON格式）

使用场景示例：

```bash
# 为模型训练准备平衡的数据集
python src/balance_dataset.py --source_dir /path/to/original_dataset --output_dir ./balanced_dataset --min_count 300 --max_count 1000

# 用平衡后的数据集训练模型
python src/train.py --data_dir ./balanced_dataset --batch_size 32 --num_epochs 20 --pretrained
```

### 提取类别名称

提取数据集中的类别名称并保存为JSON文件：

```bash
python src/extract_class_names.py --data_dir ./data --output_path ./class_names.json
```

参数说明：
- `--data_dir`: 数据集根目录（包含train, val, test子目录）
- `--output_path`: 输出文件路径（默认为`./class_names.json`）

## 模型训练

使用`train.py`脚本训练模型：

```bash
python src/train.py --data_dir ./data --batch_size 32 --num_epochs 20 --learning_rate 0.001 --model_save_dir ./models --pretrained
```

参数说明：
- `--data_dir`: 数据集根目录（默认为`./data`）
- `--batch_size`: 批次大小（默认为32）
- `--num_workers`: 数据加载器的工作线程数（默认为4）
- `--num_epochs`: 训练轮数（默认为20）
- `--learning_rate`: 学习率（默认为0.001）
- `--model_save_dir`: 模型保存路径（默认为`./models`）
- `--pretrained`: 是否使用预训练权重（默认不使用）

训练过程中会保存最佳模型和最终模型，同时在`model_save_dir`目录下生成损失曲线和准确率曲线图。

## 模型评估

使用`evaluate.py`脚本评估模型：

```bash
python src/evaluate.py --model_path ./models/best_model.pth --data_dir ./data/test --output_dir ./results
```

参数说明：
- `--model_path`: 模型路径（必需）
- `--data_dir`: 测试数据目录（必需）
- `--batch_size`: 批次大小（默认为32）
- `--num_workers`: 数据加载器的工作线程数（默认为4）
- `--output_dir`: 输出目录（默认为`./results`）

评估结果包括准确率、分类报告、混淆矩阵和预测可视化。

## 图像预测

使用`predict.py`脚本对新图像进行预测：

```bash
python src/predict.py --model_path ./models/best_model.pth --image_path /path/to/image.jpg --class_names_path ./class_names.json --output_dir ./predictions
```

参数说明：
- `--model_path`: 模型路径（必需）
- `--image_path`: 要预测的图像路径，可以是单张图像或目录（必需）
- `--class_names_path`: 类别名称JSON文件路径（可选）
- `--output_dir`: 输出目录（默认为`./predictions`）
- `--topk`: 显示前k个预测结果（默认为3）

## 示例流程

完整的项目流程示例：

1. 数据集准备与处理（选择以下流程之一）：

   基本划分流程：
   ```bash
   # 划分数据集
   python src/prepare_data.py --source_dir /path/to/dataset --output_dir ./data
   ```
   
   或使用自定义划分：
   ```bash
   python src/split_dataset.py --source_dir /path/to/dataset --output_dir ./data --splits "train:0.7,val:0.15,test:0.15"
   ```
   
   或使用数据集平衡工具：
   ```bash
   # 先划分数据集
   python src/split_dataset.py --source_dir /path/to/dataset --output_dir ./split_data --splits "train:0.7,val:0.15,test:0.15"
   
   # 然后平衡每个划分中的类别
   python src/balance_dataset.py --source_dir ./split_data/train --output_dir ./balanced_data/train --min_count 200 --max_count 2000
   python src/balance_dataset.py --source_dir ./split_data/val --output_dir ./balanced_data/val --min_count 50 --max_count 500
   python src/balance_dataset.py --source_dir ./split_data/test --output_dir ./balanced_data/test --min_count 50 --max_count 500
   ```

2. 提取类别名称：

```bash
python src/extract_class_names.py --data_dir ./data --output_path ./class_names.json
```

3. 训练模型：

```bash
python src/train.py --data_dir ./data --batch_size 32 --num_epochs 20 --pretrained
```

4. 评估模型：

```bash
python src/evaluate.py --model_path ./models/best_model.pth --data_dir ./data/test --output_dir ./results
```

5. 预测新图像：

```bash
python src/predict.py --model_path ./models/best_model.pth --image_path ./test_images --class_names_path ./class_names.json
```

## 注意事项

- 对于大型数据集，可能需要调整`--batch_size`和`--num_workers`参数。
- 如果出现内存不足错误，请尝试减小批次大小。
- 对于复杂任务，可能需要增加`--num_epochs`参数来延长训练时间。
- GPU训练会显著提高速度，如果可用，系统会自动使用。
- 在Windows系统上，建议将`--num_workers`设置为0或较小的值。
- 使用`split_dataset.py`的`symlink`模式可以节省磁盘空间，但在Windows上可能需要管理员权限。
- 数据集平衡工具对于处理不平衡数据集非常有用，可以提高模型训练效果。 