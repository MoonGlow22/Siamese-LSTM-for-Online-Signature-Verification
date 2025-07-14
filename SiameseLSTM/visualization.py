import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from config import device, OUTPUT_DIRECTORY

def plot_results(train_losses, val_losses, y_test, distances, predictions, test_persons, threshold):

    plt.figure(figsize=(15, 10))
    plt.suptitle(test_persons, fontsize=14, fontweight='bold')

    # Loss graph
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Distance distribution
    plt.subplot(2, 2, 2)
    same_distances = distances[y_test == 0]
    diff_distances = distances[y_test == 1]

    plt.hist(same_distances, bins=30, alpha=0.7, label='Same Person', color='green')
    plt.hist(diff_distances, bins=30, alpha=0.7, label='Different Person', color='red')
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()

    # ROC Curve
    plt.subplot(2, 2, 3)
    fpr, tpr, _ = roc_curve(y_test, distances)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Confusion Matrix
    plt.subplot(2, 2, 4)
    ax = plt.gca()
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")

    filename = "_".join(test_persons) + ".png" if isinstance(test_persons, list) else str(test_persons) + ".png"
    save_path = os.path.join(OUTPUT_DIRECTORY,"test_results", filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)


def plot_training_distance_distribution(model, train_loader, combination_idx):
    """
    Plots the distance distribution for training data and calculates threshold.
    
    Args:
        model: Trained Siamese LSTM model
        train_loader: DataLoader for training data
        combination_idx: Index of the current combination for file naming
    
    Returns:
        threshold: Calculated threshold value
    """
    print("\nCalculating distance distribution on training data...")
    model.eval()
    same_distances = []
    diff_distances = []

    with torch.no_grad():
        for sig1, sig2, labels in train_loader:
            sig1, sig2, labels = sig1.to(device), sig2.to(device), labels.to(device)
            distances = model(sig1, sig2)
            distances = distances.cpu().numpy()
            labels = labels.cpu().numpy()
            same_distances.extend(distances[labels == 0])
            diff_distances.extend(distances[labels == 1])

    threshold = (np.mean(same_distances) + np.mean(diff_distances)) / 2

    plt.figure(figsize=(10, 6))
    plt.hist(same_distances, bins=30, alpha=0.7, label='Same Person', color='green')
    plt.hist(diff_distances, bins=30, alpha=0.7, label='Different Person', color='red')
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Training Data - Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIRECTORY,"train_results", f'distance_distribution_combination_{combination_idx + 1}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Same person distances - Mean: {np.mean(same_distances):.4f}, Std: {np.std(same_distances):.4f}")
    print(f"Different person distances - Mean: {np.mean(diff_distances):.4f}, Std: {np.std(diff_distances):.4f}")
    print(f"Calculated threshold: {threshold:.4f}")
    
    return threshold
