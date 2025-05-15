
# 📚 PyTorch `torchvision` Classification Models (Complete Reference)

> ✅ All models support transfer learning via `pretrained=True` or `weights=Pretrained_Weights`.  
> 🧠 Default pre-training is on **ImageNet-1K (1000 classes)**.

---

## 📑 Table of Contents

1. [🧱 Classic CNN Architectures](#1-🧱-classic-cnn-architectures)  
2. [📱 Lightweight/Mobile Models](#2-📱-lightweightmobile-models)  
3. [🧠 Transformer-Based Models](#3-🧠-transformer-based-models)  
4. [⚙️ Usage Guide](#4-⚙️-usage-guide)  
5. [📥 Input Specifications](#5-📥-input-specifications)

---

## 1. 🧱 Classic CNN Architectures

### 🔹 AlexNet (2012)
```python
models.alexnet(pretrained=False)
```

---

### 🔹 VGG Family
**Standard Versions:**
```python
models.vgg11(pretrained=False)
models.vgg13(pretrained=False)
models.vgg16(pretrained=False)
models.vgg19(pretrained=False)
```

**BatchNorm Variants:**
```python
models.vgg11_bn(pretrained=False)
models.vgg16_bn(pretrained=False)
models.vgg19_bn(pretrained=False)
```

---

### 🔹 ResNet Family
**Original ResNet:**
```python
models.resnet18(pretrained=False)
models.resnet34(pretrained=False)
models.resnet50(pretrained=False)
models.resnet101(pretrained=False)
models.resnet152(pretrained=False)
```

**Enhanced Variants:**
```python
models.resnext50_32x4d(pretrained=False)
models.resnext101_32x8d(pretrained=False)
models.wide_resnet50_2(pretrained=False)
models.wide_resnet101_2(pretrained=False)
```

---

### 🔹 Inception Series
```python
models.googlenet(pretrained=False)      # Inception v1
models.inception_v3(pretrained=False)
```

---

### 🔹 DenseNet Family
```python
models.densenet121(pretrained=False)
models.densenet161(pretrained=False)
models.densenet169(pretrained=False)
models.densenet201(pretrained=False)
```

---

## 2. 📱 Lightweight/Mobile Models

### 🔸 MobileNet Series
```python
models.mobilenet_v2(pretrained=False)
models.mobilenet_v3_large(pretrained=False)
models.mobilenet_v3_small(pretrained=False)
```

### 🔸 ShuffleNet
```python
models.shufflenet_v2_x0_5(pretrained=False)   # 0.5x complexity
models.shufflenet_v2_x1_0(pretrained=False)   # 1.0x complexity
```

### 🔸 EfficientNet (B0–B7)
```python
models.efficientnet_b0(pretrained=False)
models.efficientnet_b1(pretrained=False)
...
models.efficientnet_b7(pretrained=False)
```

### 🔸 MNASNet
```python
models.mnasnet0_5(pretrained=False)
models.mnasnet1_0(pretrained=False)
```

### 🔸 RegNet Family
**RegNetY:**
```python
models.regnet_y_400mf(pretrained=False)
models.regnet_y_8gf(pretrained=False)
```

**RegNetX:**
```python
models.regnet_x_400mf(pretrained=False)
models.regnet_x_1_6gf(pretrained=False)
```

---

## 3. 🧠 Transformer-Based Models

### 🔹 Vision Transformer (ViT)
```python
models.vit_b_16(pretrained=False)
models.vit_b_32(pretrained=False)
models.vit_l_16(pretrained=False)
```

### 🔹 Swin Transformer
```python
models.swin_t(pretrained=False)  # Tiny
models.swin_s(pretrained=False)  # Small
models.swin_b(pretrained=False)  # Base
```

---

## 4. ⚙️ Usage Guide

### ✅ Load Pretrained Model
```python
from torchvision import models

model = models.resnet50(pretrained=True)  # or use weights=
```

### 🔧 Modify for Transfer Learning
```python
# Replace final layer
num_classes = 5
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
```

---

## 5. 📥 Input Specifications

| Model Family       | Input Size      | Normalization (mean, std)               |
|--------------------|------------------|-----------------------------------------|
| All (ImageNet)     | (3, 224, 224)     | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Inception v3       | (3, 299, 299)     | Same normalization                      |
