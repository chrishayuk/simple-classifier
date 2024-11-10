import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from jsonl_dataset import JSONLDataset
from sklearn.metrics import accuracy_score
from simple_binary_classifier import SimpleClassifier
from evaluate import evaluate

# Model training function
def train(model, dataloader, loss_function, optimizer, num_epochs):
    # Loop through each epoch
    for epoch in range(num_epochs):
        all_labels = []
        all_predictions = []
        
        # Loop through the dataset
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Collect predictions and labels for accuracy calculation
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
        
        # Calculate and print accuracy for this epoch
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Argument parsing function
def parse_args():
    # setup the parser
    parser = argparse.ArgumentParser(description="Train and evaluate a binary classifier on structured JSONL data")

    # set arguments
    parser.add_argument('--train_file', type=str, default='output/train_data.jsonl', help="Path to training data JSONL file")
    parser.add_argument('--test_file', type=str, default='output/test_data.jsonl', help="Path to test data JSONL file")
    parser.add_argument('--input_size', type=int, default=10, help="Number of input features")
    parser.add_argument('--hidden_size', type=int, default=5, help="Number of hidden layer units")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for DataLoader")
    parser.add_argument('--model_path', type=str, default='output/model.pth', help="Path to save the trained model")

    # parse
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the training and test datasets
    train_dataset = JSONLDataset(args.train_file)
    test_dataset = JSONLDataset(args.test_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SimpleClassifier(args.input_size, args.hidden_size, num_classes=2)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    print("Starting training...")
    train(model, train_dataloader, loss_function, optimizer, args.num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    # Evaluate the model
    print("\nEvaluating model on test data...")
    evaluate(model, test_dataloader)

if __name__ == "__main__":
    main()
