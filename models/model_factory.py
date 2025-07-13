from torchvision import models
from torch import nn
import torch
from models.custom_cnn import ConvBaseline


def get_model(name, in_channels = 1, num_classes = 4):
    """
        Returns a model instance based on the given name.

        Args:
            name (str): Model name ('cnn_baseline' or 'resnet18').
            in_channels (int): Number of input channels. Used for 'cnn_baseline'.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights (only for 'resnet18').

        Returns:
            torch.nn.Module: Instantiated model.
        """
    if name == "cnn_baseline":
        return ConvBaseline(in_channels, num_classes)
    elif name == "resnet50":

        model = models.resnet50(weights = "IMAGENET1K_V2")
        model.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=model.conv1.out_channels, kernel_size = model.conv1.kernel_size, stride= model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
        model.fc = nn.Linear(model.fc.in_features, out_features=num_classes, bias=True)
        return model
    
    elif name == "resnet18":
        model = models.resnet18(weights = "IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                                padding=model.conv1.padding, bias=model.conv1.bias)
        model.fc = nn.Linear(model.fc.in_features, out_features=num_classes, bias=True)
        return model

    else:
        raise ValueError(f"Model {name} not recognized")


def save_weights(model: nn.Module, filename: str ="weights.pth"):
    torch.save(model.state_dict(), filename)

def load_weights(model: nn.Module, filename: str ="weights.pth"):
    model.load_state_dict(torch.load(filename))
    return model

