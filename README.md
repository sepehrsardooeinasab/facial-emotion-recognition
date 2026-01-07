# FER+ Facial Expression Recognition Benchmark  
**ResNet50 · DDAMFN · ResEmoteNet (PyTorch)**

This project benchmarks three deep learning architectures on the **FER+ (FER2013Plus)** dataset using **identical image preprocessing and training protocol**.  
The goal is a **fair comparison** of model performance for facial expression recognition.

---

## Dataset

**FER2013Plus (FER+)**
- Source: https://www.kaggle.com/datasets/subhaditya/fer2013plus
- Grayscale facial images
- Resolution: 48 × 48
- 8 emotion classes (as defined in FER+)

---

## Models Evaluated

- **ResNet50 (Pretrained)**  
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

- **DDAMFN (Dual Dynamic Attention Model with Feature Normalization)**  
  https://github.com/SainingZhang/DDAMFN/tree/main

- **ResEmoteNet**  
  https://github.com/ArnabKumarRoy02/ResEmoteNet

All models are implemented and trained using **PyTorch**.

---

## Training Setup

- Framework: PyTorch
- Loss Function: Cross-Entropy Loss
- Learning Rate: configurable per experiment
- Batch Size: configurable
- Train / Test split: consistent across all models

---

## Performance Results

**Final Train and Test Accuracy**

| Model | Train Accuracy (%) | Test Accuracy (%) |
|------|-------------------:|------------------:|
| **ResNet50 (Pretrained)** | **97.41** | **83.77** |
| **DDAMFN** | 94.80 | 81.53 |
| **ResEmoteNet** | 94.99 | 80.63 |

> ResNet50 achieves the highest generalization performance on FER+, while DDAMFN and ResEmoteNet provide competitive results with lighter or specialized architectures.

---

## Training Curves

The repository includes:
- Per-epoch **training and validation loss**
- Per-epoch **training and validation accuracy**

Location:
```
outputs/figures/
├─ resnet50_loss_accuracy.png
├─ ddamfn_loss_accuracy.png
└─ resemotenet_loss_accuracy.png
```



---

## Model Architectures

Architecture diagrams for all three models are provided:
```
outputs/figures/
├─ resnet50_architecture.png
├─ ddamfn_architecture.png
└─ resemotenet_architecture.png
```


---

## Example FER+ Images

Example FER+ input images and corresponding facial expressions are available in:
```
outputs/figures/
└─ ferplus_examples.png
```