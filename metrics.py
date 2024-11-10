# metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.utils import resample

def display_confusion_matrix(true_labels, predicted_labels):
    """
    Generates and displays a confusion matrix with labeled rows and columns.
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual: 0", "Actual: 1"],
        columns=["Predicted: 0", "Predicted: 1"]
    )

    # Pretty print the confusion matrix using tabulate
    print("Confusion Matrix:")
    print(tabulate(conf_matrix_df, headers="keys", tablefmt="grid"))

def calculate_balanced_accuracy_25th(true_labels, predicted_labels, n_splits=5):
    """
    Calculates the 25th percentile of Balanced Accuracy by simulating K-fold cross-validation.
    """
    # Reshape data into n_splits subsets
    subset_size = len(true_labels) // n_splits
    balanced_accuracies = []

    for i in range(n_splits):
        # Create bootstrapped samples to mimic cross-validation
        subset_true, subset_pred = resample(true_labels, predicted_labels, n_samples=subset_size, random_state=i)
        # Calculate balanced accuracy for each subset
        balanced_accuracy = balanced_accuracy_score(subset_true, subset_pred)
        balanced_accuracies.append(balanced_accuracy)

    # Calculate the 25th percentile of balanced accuracies
    return np.percentile(balanced_accuracies, 25)

def calculate_diversity_score(true_labels, n_clusters=10):
    """
    Estimates a diversity score based on label distributions in simulated clusters.
    """
    # Simulate clusters as equal divisions of label data
    labels_array = np.array(true_labels)
    diversity_counts = np.zeros(n_clusters)

    for i in range(n_clusters):
        # Count positive and negative labels in each cluster
        cluster_labels = labels_array[i::n_clusters]
        unique, counts = np.unique(cluster_labels, return_counts=True)
        # Sum counts for each unique label, capping to a max value
        diversity_counts[i] = min(sum(counts), 10)  # Cap at 10 for normalization

    # Normalize diversity score between 0 and 1
    return diversity_counts.sum() / (2 * n_clusters * 10)

def display_classification_metrics(true_labels, predicted_labels):
    """
    Calculates and displays classification metrics including accuracy, precision, recall, F1 score,
    balanced accuracy (25th percentile), and diversity score.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    
    # Calculate additional metrics
    balanced_accuracy_25th = calculate_balanced_accuracy_25th(true_labels, predicted_labels)
    diversity_score = calculate_diversity_score(true_labels)

    # Display the calculated metrics as a formatted table
    metrics_table = [
        ["Accuracy", f"{accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"],
        ["Balanced Accuracy (25th percentile)", f"{balanced_accuracy_25th:.4f}"],
        ["Diversity Score", f"{diversity_score:.4f}"]
    ]

    print("\nMetrics:")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
