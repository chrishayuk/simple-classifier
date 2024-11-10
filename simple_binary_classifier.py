import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers=5, dropout_prob=0.1):
        # call the constructor
        super(SimpleClassifier, self).__init__()

        # Dynamically create hidden layers based on num_hidden_layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_size))

        # add the activation function, after the first layer
        layers.append(nn.SiLU())

        # add a dropout layer
        layers.append(nn.Dropout(dropout_prob))

        # Add the remaining hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, num_classes))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the entire model
        return self.model(x)
