import torch
import json
import argparse
import os
from sklearn.model_selection import train_test_split

def generate_data(num_samples=1000, input_size=10):
    # Initialize an empty list to store generated data
    data = []
    
    # Loop through the number of samples to generate data
    for _ in range(num_samples):
        # Generate random feature data as a list
        x_data = torch.randn(input_size).tolist()
        
        # Apply a rule for label: if sum of first half > sum of second half, label = 1, else 0
        y_data = int(sum(x_data[:input_size // 2]) > sum(x_data[input_size // 2:]))
        
        # Append the data as a dictionary to the list
        data.append({'features': x_data, 'label': y_data})
    
    # Return the generated data
    return data

def save_to_jsonl(data, output_file):
    # Create the output directory if it doesn’t exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Loop through each entry in the data
        for entry in data:
            # Write each entry as a JSON object followed by a newline
            f.write(json.dumps(entry) + '\n')
    
    # Print confirmation message with the number of samples saved
    print(f"Saved {len(data)} samples to {output_file}")

if __name__ == "__main__":
    # Set up argument parsing for command-line flexibility
    parser = argparse.ArgumentParser(description="Generate synthetic data for binary classification")

    # set arguments
    parser.add_argument('--num_samples', type=int, default=1000, help="Total number of samples to generate")
    parser.add_argument('--input_size', type=int, default=10, help="Number of features in each sample")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save generated JSONL files")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Train-test split ratio")

    # parse arguments
    args = parser.parse_args()

    # Generate structured data based on a simple rule
    data = generate_data(num_samples=args.num_samples, input_size=args.input_size)

    # Split into train and test sets based on the split ratio
    train_data, test_data = train_test_split(data, test_size=(1 - args.split_ratio), random_state=42)

    # Save train and test sets to separate files in the specified output directory
    save_to_jsonl(train_data, output_file=os.path.join(args.output_dir, 'train_data.jsonl'))
    save_to_jsonl(test_data, output_file=os.path.join(args.output_dir, 'test_data.jsonl'))
