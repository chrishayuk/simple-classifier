import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def display_confusion_matrix(true_labels, predicted_labels):
    """
    Generates and displays a confusion matrix with labeled rows and columns.

    Parameters:
    - true_labels: List of true class labels.
    - predicted_labels: List of predicted class labels.
    """
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Convert confusion matrix to a labeled DataFrame for readability
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual: 0", "Actual: 1"],
        columns=["Predicted: 0", "Predicted: 1"]
    )

    # Display the labeled confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix_df)

def display_classification_metrics(true_labels, predicted_labels):
    """
    Calculates and displays classification metrics including accuracy, precision, recall, and F1 score.

    Parameters:
    - true_labels: List of true class labels.
    - predicted_labels: List of predicted class labels.
    """
    # Calculate classification metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')

    # Display the calculated metrics
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
