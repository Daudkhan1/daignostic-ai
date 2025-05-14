import numpy as np
import torch
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from torchsummary import summary
from tqdm import tqdm


def compute_class_weights(num_classes, labels, device):
    """
    Compute class weights to handle class imbalance in the dataset.
    Args:
        num_classes (int): Total number of unique classes
        labels (tensor or numpy array): 1D array of class labels
        device (torch.device): The device (e.g., 'cuda' or 'cpu')
                               to which the class weights will be moved.
    Returns:
        class_weights (tensor): Tensor of class weights
    """
    # Compute class weights using scikit-learn utility
    class_weights = compute_class_weight("balanced", classes=num_classes, y=labels)
    # 1. Helps reduce bias towards frequent classes
    # 2. Used in loss function => criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def get_class_weights_for_loader(loader, num_classes, device):
    """
    Get class weights based on the class distribution in a DataLoader.
    Args:
        loader (DataLoader): DataLoader for the dataset (train/validation)
        num_classes (int): Number of classes
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to which
                               the class weights will be moved
    Returns:
        class_weights (tensor): Tensor of class weights
    """
    all_labels = []
    for _, labels in tqdm(loader, desc="Processing Batches"):
        all_labels.extend(labels.numpy())

    # print ("all_labels_are", all_labels)

    classes = np.array(sorted(np.unique(all_labels)))

    print("classes_are", classes)

    # Compute class weights
    class_weights = compute_class_weights(classes, all_labels, device)

    return class_weights


def log_model_summary(model, input_size=(3, 224, 224), writer=None):
    """Logs model architecture summary to TensorBoard or console"""
    if writer:
        summary_str = summary(
            model,
            input_size=input_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        writer.add_text("Model Summary", summary_str)
    else:
        print("Model Summary:\n")
        summary(
            model,
            input_size=input_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
