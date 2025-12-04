# 🧠 Brain Tumor Segmentation Using U-Net

This project implements a **U-Net convolutional neural network** to perform **binary semantic segmentation** of brain tumors in MRI images.  
Accurate segmentation is essential for diagnosis, treatment planning, and patient monitoring. Using a Kaggle dataset of **2146 annotated MRI images**, this project builds a complete pipeline from preprocessing → mask extraction → model training → evaluation.

---

## 🚀 Project Features

- Uses **U-Net**, the standard architecture for biomedical image segmentation  
- Performs segmentation of **tumor vs. non-tumor** regions  
- Extracts masks from **COCO-format annotations**  
- Full preprocessing pipeline:
  - Auto-orientation  
  - Mask extraction  
  - Resizing (128×128)  
  - Normalization  
- Trains a U-Net with:
  - Dice Loss  
  - Adam optimizer  
  - 50 epochs  
- Includes evaluation using Dice, F1, Precision, Recall, Jaccard  
- Reproducible PyTorch/Keras model code in notebook form

---


## 📊 Dataset

- **Source:** Kaggle — Brain Tumor Image Semantic Segmentation  
- **Link:** https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation  
- **Total Images:** 2146  
- **Annotation Format:** COCO  
- **Classes:**  
  - `0` — Non-tumor  
  - `1` — Tumor  

### Preprocessing Steps
- Extract segmentation polygons from COCO JSON  
- Convert to binary masks  
- Resize images + masks to **128 × 128**  
- Normalize images to `[0,1]`  
- Split into train/validation/test sets  

---

## 🧠 Model Architecture — U-Net

This project implements a **standard U-Net** composed of:

### **1. Encoder**
- 4 convolutional blocks  
- Each block followed by **2×2 max pooling**  
- Feature depth doubles at each stage  

### **2. Bottleneck**
- Deepest layer with **1024 filters**  
- Captures highest-level features  

### **3. Decoder**
- 4 up-sampling blocks  
- Each uses:
  - Transposed convolution  
  - Skip connections (concatenate with encoder)  
  - Convolution layers to refine upsampled maps  

### **4. Output Layer**
1×1 convolution
Sigmoid activation
Binary segmentation mask (0–1)

---

## ⚙️ Training Details

### **Hyperparameters**

| Parameter | Value |
|----------|--------|
| Epochs | 50 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | Dice Loss |
| Batch Size | 16 (if applicable) |

### **Loss Function**

The **Dice Loss** is ideal for segmentation tasks with class imbalance.  
A small smoothing term (`1e-15`) is included for numerical stability.

---

## 📉 Results

The model achieves strong segmentation accuracy on the test set:

| Metric | Score |
|--------|-------|
| Test Loss | **0.249** |
| Dice Coefficient | **0.744** |
| F1 Score | **0.758** |
| Precision | **0.832** |
| Recall | **0.696** |
| Jaccard Index | **0.610** |

### Interpretation
- **Precision (0.832)** is high → few false positives  
- **Recall (0.696)** shows some tumor regions may be missed → room for improvement  
- **Dice = 0.744** → strong overall segmentation performance  

---

## 🧩 Future Work

- Experiment with:
  - Attention U-Net  
  - Residual U-Net  
  - Dense U-Net  
  - 3D U-Net models  
- Add data augmentation (rotation, contrast, elastic deform)  
- Train with higher-resolution images (e.g., 256×256 or 512×512)  
- Use Tversky loss to reduce false negatives  

---

## 📚 References

1. Saluja et al., *U-Net Variants for Brain Tumor Segmentation*, 2023  
2. Zhang et al., *Shuffle Attention Residual U-Net*, 2022  
3. Abraham et al., *Deep Learning for Ultrasound Segmentation*, 2019  
4. Kaggle: *Brain Tumor Image Dataset*, 2024  
5. Chen et al., *3D U-Net for Medical Image Segmentation*, 2020  
6. Yang & Song, *Automatic Brain Tumor Segmentation*, 2018  
7. Liu et al., *Segmentation via Knowledge Transfer*, 2023  
8. Reswave-Net, Wavelet-based U-Net, 2023  
9. Mathews & Mohamed, U-Net Review, 2020  
10. Goni et al., Attention-Sharp-U-Net, 2022  
11. Kong & Zhang, *Multimodal Cascaded U-Net*, 2021  

---

## ✨ Author

**Safwan Hasan**  
Department of Electrical and Computer Engineering  
Toronto Metropolitan University

