import torch
import torch.nn as nn

# Define the classifier model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # call the constructor
        super(SimpleClassifier, self).__init__()

        # 2 layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)

        # relu activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # forward pass through the first layer
        out = self.layer1(x)

        # apply the activation function
        out = self.relu(out)

        # forward pass through the second layer
        out = self.layer2(out)

        # return the result
        return out