import time
import torch
from sklearn.metrics import f1_score

def train_epoch(model, dataloader, criterion, optimizer, device):

    """
        Trains the model for one epoch on the given data.

        Args:
            model (torch.nn.Module): The neural network model.
            dataloader (DataLoader): DataLoader for the training data.
            criterion (loss): Loss function.
            optimizer (Optimizer): Optimizer for updating model weights.
            device (torch.device): Device to run the computations on.

        Returns:
            tuple: Average loss and accuracy for the epoch.
        """

    model.train()
    running_loss, running_corrects, total = 0.0, 0.0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    loss = running_loss / total
    acc = running_corrects.double() / total
    return loss, acc

def evaluate(model, dataloader, criterion, device):

    """
        Evaluates the model on the given dataset.

        Args:
            model (torch.nn.Module): The neural network model.
            dataloader (DataLoader): DataLoader for the evaluation data.
            criterion (loss): Loss function.
            device (torch.device): Device to run the computations on.

        Returns:
            tuple: Average loss, accuracy, and F1 score.
        """

    model.eval()
    running_loss, running_corrects, total = 0.0, 0.0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / total
    acc = running_corrects.double() / total
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return loss, acc, f1


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs):

    """
       Trains the model for a specified number of epochs, evaluating on the test set after each epoch.

       Args:
           model (torch.nn.Module): The neural network model.
           train_loader (DataLoader): DataLoader for the training data.
           test_loader (DataLoader): DataLoader for the testing data.
           criterion (loss): Loss function.
           optimizer (Optimizer): Optimizer for model training.
           scheduler (lr_scheduler, optional): Learning rate scheduler. Can be None.
           device (torch.device): Device for computation.
           num_epochs (int): Number of training epochs.

       Returns:
           torch.nn.Module: The trained model.
       """

    model = model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

        if scheduler:
            scheduler.step()

        elapsed_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed_time:.2f}s ")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"    Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    return model