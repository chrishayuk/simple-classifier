import argparse
import os
import torch
from torch.utils.data import DataLoader
from jsonl_dataset import JSONLDataset
from simple_binary_classifier import SimpleClassifier
from metrics import display_confusion_matrix, display_classification_metrics

def evaluate(model, dataloader):
    """
    Evaluates the model on the provided dataloader, displaying the confusion matrix
    and key metrics (accuracy, precision, recall, and F1 score).
    """
    true_labels = []
    predicted_labels = []

    # Set model to evaluation mode
    model.eval()

    # Collect predictions and true labels
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    # Display the confusion matrix and metrics using functions from metrics.py
    display_confusion_matrix(true_labels, predicted_labels)
    display_classification_metrics(true_labels, predicted_labels)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained binary classifier on a test dataset")

    parser.add_argument('--model_path', type=str, default='output/model.pth', help="Path to the saved model file (.pth)")
    parser.add_argument('--test_file', type=str, default='output/test_data.jsonl', help="Path to the test data JSONL file")
    parser.add_argument('--input_size', type=int, default=10, help="Number of input features")
    parser.add_argument('--hidden_size', type=int, default=64, help="Number of hidden layer units")
    parser.add_argument('--num_hidden_layers', type=int, default=7, help="Number of hidden layers in the model")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for DataLoader")

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the test dataset
    test_dataset = JSONLDataset(args.test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model with specified input and hidden layer sizes
    model = SimpleClassifier(
        input_size=args.input_size, 
        hidden_size=args.hidden_size, 
        num_classes=2, 
        num_hidden_layers=args.num_hidden_layers
    )

    # Load model weights
    if os.path.exists(args.model_path):
        # load the model weights
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        print(f"Loaded model from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    

    # Evaluate the model on the test dataset
    evaluate(model, test_dataloader)

if __name__ == "__main__":
    main()
