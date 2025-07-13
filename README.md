# Chest X-ray Classification (COVID-19, Pneumonia, Lung Opacity, Normal)

This project focuses on classifying chest X-ray images into four classes: **COVID-19**, **Viral Pneumonia**, **Lung Opacity**, and **Normal**. It explores deep learning techniques under the challenges of medical image classification and class imbalance.

---

## 🗂 Dataset

* Source: [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)
* \~19,000 X-ray images (unbalanced across classes)
* Includes segmentation masks for lungs (used for region-focused training)

### Preprocessing:

* Applied lung segmentation masks to focus on regions of interest
* Data augmentations: mild random rotations and flips

---

## 🧠 Models

### 1. **CNN Baseline**

* Custom convolutional network trained from scratch
* Used as a performance baseline

### 2. **ResNet18** (Transfer Learning)

* Pretrained on ImageNet
* Fully fine-tuned
* Compared with and without class imbalance strategies

---

## ⚖️ Class Imbalance Strategies

1. **None (Baseline)**
2. **Weighted Loss**: Class weights inversely proportional to frequency
3. **Oversampling**: `WeightedRandomSampler` to balance batches

---

## 📊 Evaluation Metrics

* **Per-class**: Precision, Recall, F1-Score
* **Macro averages** across all classes
* **Accuracy** (for reference)
* **Confusion matrix**
* **ROC and Precision-Recall curves** (per class)

---

## ✅ Results (Macro F1-Score)

| Model        | Weighted Loss | Oversampling | Macro F1 |
| ------------ | ------------- | ------------ | -------- |
| CNN Baseline | No            | No           | 0.80     |
| ResNet18     | No            | No           | 0.89     |
| ResNet18     | Yes           | No           | 0.86     |
| ResNet18     | No            | Yes          | 0.90     |

---

## 🏁 Conclusion

* Custom CNN provides a reasonable starting point.
* Transfer learning with ResNet18 outperforms it.
* Oversampling gave the best results in this setup.
* Class Imbalance Strategies was not that effective (minor class had already good scoring)

---


## 🛠️ Requirements

```bash
python >= 3.8
torch >= 1.10
torchvision
scikit-learn
matplotlib
numpy
pandas
```

---

## 📁 Structure

```
├── models/                # CNN baseline architecture + save/load model functions
├── notebooks/             # Training and results with visualizations
├── outputs/               # Saved curves and confusion matrix as .png files
├── preprocessing/         # Creating DataLoaders, Data Augmentation, Class Imbalance Strategies functions
├── results/               # Evaluation + Visualizations functions
├── training/              # Training loops
├── utils/                 # Other helper functions
├── weights/               # Saved model's weights
├── main.py                # Pipeline to train and evaluate model

```
