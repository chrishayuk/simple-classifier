import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from jsonl_dataset import JSONLDataset
from simple_binary_classifier import SimpleClassifier

# number of features in the input
input_size = 10  

# size of hidden layer
hidden_size = 5  

# binary classification (class 0 and class 1)
num_classes = 2  

# learning rate
learning_rate = 0.0001

# Training loop
num_epochs = 100

# Load the dataset from JSONL file and create DataLoader
dataset = JSONLDataset('output/data.jsonl')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleClassifier(input_size, hidden_size, num_classes)

# set the loss function as cross entropy loss
loss_function = nn.CrossEntropyLoss()

# set the optimizer as an adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# perform each epoch
for epoch in range(num_epochs):
    # get the inputs and labels from the dataset
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)

        # apply the loss function
        loss = loss_function(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # output result
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    # Generate random test data (10 samples) to evaluate the model
    test_data = torch.randn(10, input_size)
    test_outputs = model(test_data)
    _, predicted = torch.max(test_outputs, 1)
    print("Predicted labels:", predicted)
