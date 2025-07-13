from models.model_factory import get_model, save_weights
from preprocessing.data import get_loaders, train_test_split, calculate_mean_std, create_datasets
from training.training import train_model
from results.evaluation import evaluate_model
from results.visualizations import plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curve, \
    show_sample_images
import torch
from torch import nn
from utils.seed import set_seed


def main():
    num_epochs = 10
    split = False
    split_ratio = 0.8
    data_path = 'data'
    batch_size = 32
    in_channels = 1
    classes = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']
    num_classes = len(classes)
    model_name = 'resnet18'
    step_size = 5
    gamma = 0.1
    learning_rate = 0.001
    seed = 42

    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if split:
        train_test_split(data_path, split_ratio=split_ratio)

    mean, std = calculate_mean_std(data_path, batch_size=batch_size)
    print(f"mean: {mean}, std: {std}")

    train_loader, test_loader = get_loaders(data_path, batch_size=batch_size, mean=mean, std=std)

    model = get_model(name = model_name, in_channels=in_channels, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    trained_model = train_model(model = model, train_loader = train_loader, test_loader = test_loader, criterion = criterion, optimizer = optimizer, scheduler = scheduler, device = device, num_epochs=num_epochs)
    save_weights(trained_model, "resnet18_v2.pth")

    results = evaluate_model(trained_model, test_loader, device = device, num_classes=num_classes)

    print("RESULTS: ")
    print("*"*30)
    print(f"Precision - macro avg: {results['precision']}")
    print(f"Recall - macro avg: {results['recall']}")
    print(f"F1-score - macro avg: {results['f1']}")
    print(f"Accuracy:  {results['acc']}")
    print("*"*30)
    print(f"Covid - metrics: {results['report']['COVID']}")
    print("*" * 30)
    print(f"Lung Opacity - metrics: {results['report']['Lung Opacity']}")
    print("*" * 30)
    print(f"Normal - metrics: {results['report']['Normal']}")
    print("*" * 30)
    print(f"Viral Pneumonia - metrics: {results['report']['Viral Pneumonia']}")

    plot_confusion_matrix(results['confusion_matrix'], class_names=classes, save_to_file=True)
    plot_roc_curves(fpr_list = results['fpr'], tpr_list = results['tpr'], roc_auc_list = results['roc_auc'], class_names=classes, save_to_file=True)
    plot_precision_recall_curve(thresholds_list=results['thresholds_pr'], precision_list=results['precision_curve'] , recall_list=results['recall_curve'], class_names=classes, save_to_file=True)






if __name__ == '__main__':
    main()