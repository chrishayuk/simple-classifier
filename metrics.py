import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tabulate import tabulate

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

def display_classification_metrics(true_labels, predicted_labels):
    """
    Calculates and displays classification metrics including accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')

    # Display the calculated metrics as a formatted table
    metrics_table = [
        ["Accuracy", f"{accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"]
    ]

    print("\nMetrics:")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
