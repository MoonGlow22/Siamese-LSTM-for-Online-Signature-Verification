import numpy as np
import torch
from config import device
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, threshold):
    """
    Evaluate model performance

    Args:
        model: Trained model
        test_loader: Test data loader
        threshold: Threshold to be used

    Returns:
        accuracy, all_distances, all_labels, predictions, precision, recall, f1
    """
    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for sig1, sig2, labels in test_loader:
            sig1, sig2, labels = sig1.to(device), sig2.to(device), labels.to(device)

            distances = model(sig1, sig2)

            all_distances.extend(distances.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    predictions = (all_distances > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Used Threshold: {threshold:.4f}")

    # Distance analysis
    same_indices = np.where(all_labels == 0)[0]
    diff_indices = np.where(all_labels == 1)[0]

    same_distances = all_distances[same_indices]
    diff_distances = all_distances[diff_indices]

    print("\nDistance Analysis:")
    print(f"Same person - Mean: {np.mean(same_distances):.4f}, Std: {np.std(same_distances):.4f}")
    print(f"Different person - Mean: {np.mean(diff_distances):.4f}, Std: {np.std(diff_distances):.4f}")

    return accuracy, all_distances, all_labels, predictions, precision, recall, f1
