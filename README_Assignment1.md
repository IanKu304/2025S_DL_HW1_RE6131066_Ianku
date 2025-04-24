
# 🧠 MILS – Assignment I Report  
**Author:** Ian Ku  
**Course:** Multimedia Analysis in Sports Technology  
**Instructor:** Prof. Chih-Chung Hsu  

---

## 🗂️ Contents
- [Dataset](#dataset)
- [Problem A: Dynamic Convolution Module](#problem-a-dynamic-convolution-module)
- [Problem B: Two-Layer Network](#problem-b-two-layer-network)
- [Result Summary](#result-summary)
- [Installation & Usage](#installation--usage)

---

## 📦 Dataset

We used the **mini-ImageNet** dataset provided in the assignment:
- Images organized into 50 classes
- Each image labeled via `train.txt`, `val.txt`, `test.txt`
- Each sample is of size `32x32` (resized or optionally `224x224` for deeper models)

---

## 🧩 Problem A: Dynamic Convolution Module

### 🔧 Task Objective
Design a **convolutional module** that:
- Can handle any number of input channels (e.g., RGB, RG, R...)
- Is spatial-size invariant
- Generates convolution kernels **dynamically** at inference time

### 📐 Architecture Design

| Model | Architecture Summary |
|-------|-----------------------|
| **DynamicCNN v1** | Single dynamic convolution with MLP-based weight generator |
| **DynamicCNN v2** | Multi-layer network using CNN-based weight generator for kernel generation |
| **BaselineCNN** | Standard Conv → BN → ReLU → Pooling x2 |

### ⚙️ Implementation Details

- Weight generator takes dummy input and generates kernel weights dynamically
- Input channels (1~3) supported with channel-drop simulation
- Models trained with and without `RandomChannelDrop`

### 🎯 Accuracy and Cost Comparison (Test Set, 32x32)

#### ⛔ Without Random Drop:

| Model         | Accuracy | FLOPs        | Params     |
|---------------|----------|--------------|------------|
| DynamicCNN v1 | 0.2422   | 180.39 KMac  | 31.57 K    |
| DynamicCNN v2 | 0.1422   | 19.49 MMac   | 63.31 K    |
| BaselineCNN   | 0.2844   | 5.95 MMac    | 22.83 K    |

#### ✅ With `RandomChannelDrop`:

| Model         | Accuracy | FLOPs        | Params     |
|---------------|----------|--------------|------------|
| DynamicCNN v1 | 0.1244   | 180.39 KMac  | 31.57 K    |
| DynamicCNN v2 | 0.2244   | 19.49 MMac   | 63.31 K    |
| BaselineCNN   | 0.2422   | 5.95 MMac    | 22.83 K    |

> 💬 *DynamicCNN v1 在沒有通道遮蔽時表現佳，但在遇到隨機通道遮蔽後性能大幅下降；Dynamic v2 在通道遮蔽情境中則表現更穩定，顯示 CNN-based kernel generator 的泛化力。*

---

## 🏗️ Problem B: Two-Layer Network

### 🎯 Task Goal

Design a network with 2–4 effective layers that achieves **≥90% of ResNet34 accuracy** trained from scratch on mini-ImageNet (resize to `224x224`).

### 📐 Models

| Model Name       | Structure Summary |
|------------------|-------------------|
| **ResNet34**     | Deep residual network (baseline reference) |
| **Simple2LayerCNN** | 2 convolutional blocks with large kernels |
| **Simple4LayerCNN** | 4 stacked conv layers, no residual |
| **SE2LayerCNN**  | Simple2LayerCNN + SE-Block channel attention |

### 🧪 Results (resize 224x224)

| Model         | Accuracy | FLOPs     | Params    | Train Time |
|---------------|----------|-----------|-----------|------------|
| ResNet34      | 0.5667   | 3.68 GMac | 21.31 M   | 234.85 min |
| Simple2Layer  | 0.4400   | 43.66 MMac| 237.75 K  | 451.15 min |
| Simple4Layer  | 0.4000   | 285.83 MMac | 301.11 K| 470.57 min |
| SE2LayerCNN   | 0.4533   | 43.82 MMac| 240.51 K  | > 449 min  |

> 💬 *雖然 Simple2LayerCNN 未達 ResNet34 的 90% 門檻（目標 0.5100），但透過注意力（SE Block）提升泛化性能。Simple4LayerCNN 增加深度後性能不升反降，顯示模型需要更佳正則化或參數調整。*

---

## 📊 Result Summary

### 🔍 Problem A

- **DynamicCNN v1** 較適合純 RGB 模式，FLOPs/Params 最小
- **DynamicCNN v2** 較能處理通道變化，具備泛化性
- **BaselineCNN** 雖為靜態結構，但在統一輸入下表現穩定最佳

### 🔍 Problem B

- **Simple2LayerCNN** 提供不錯的效能與效能密度
- **SE2LayerCNN** 提升準確率並未大幅增加計算成本
- **Simple4LayerCNN** 顯示增加深度未必直接帶來改善

---

## 🛠 Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

請確保 `images/`, `train.txt`, `val.txt`, `test.txt` 放置正確，並參考 `main.py` 執行各模型訓練與測試。

---

## 📁 Additional Notes

- 可參考 `training_curves.png` 查看每模型的 loss/accuracy 變化
- 所有實驗均已紀錄 `outlier_channels.txt`，可查看非三通道處理情況
