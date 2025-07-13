import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
import seaborn as sns

def show_sample_images(dataset, num_images=5):
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        image, label = dataset[i]

        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)

        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_roc_curves(fpr_list, tpr_list, roc_auc_list, class_names=None, save_to_file=False):
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(len(fpr_list))]

    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5), sharey=True)

    for i in range(num_classes):
        ax = axes[i]
        fpr = fpr_list[i]
        tpr = tpr_list[i]
        auc = roc_auc_list[i]

        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        if i == 0:
            ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC - {class_names[i]}')
        ax.legend(loc="lower right")
        ax.grid(True)

    plt.tight_layout()

    if save_to_file:
        plt.savefig('outputs/roc_curves.png')
    plt.show()


import matplotlib.pyplot as plt

def plot_precision_recall_curve(thresholds_list, precision_list, recall_list, threshold_list=None, class_names=None, save_to_file=False):
    if class_names is None:
        class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5), sharey=True)

    for i in range(num_classes):
        ax = axes[i]
        thresholds = thresholds_list[i]
        precision = precision_list[i]
        recall = recall_list[i]

        ax.plot(thresholds, precision[:-1], label='Precision')
        ax.plot(thresholds, recall[:-1], label='Recall')

        if threshold_list is not None:
            ax.axvline(x=threshold_list[i], color='red', linestyle='--', label=f'Th = {threshold_list[i]:.2f}')

        ax.set_xlabel('Threshold')
        ax.set_title(class_names[i])
        ax.grid(True)
        if i == 0:
            ax.set_ylabel('Score')
        ax.legend()

    plt.tight_layout()

    if save_to_file:
        plt.savefig('outputs/pr_curve.png')
    plt.show()




def plot_confusion_matrix(cm ,class_names, save_to_file = False):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')


    if save_to_file:
        plt.savefig('outputs/confusion_matrix.png')
    plt.show()
