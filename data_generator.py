import torch
import json

def generate_data(num_samples=100, input_size=10):
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

def save_to_jsonl(data, output_file='output/data.jsonl'):
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Loop through each entry in the data
        for entry in data:
            # Write each entry as a JSON object followed by a newline
            f.write(json.dumps(entry) + '\n')
    
    # Print confirmation message with number of samples saved
    print(f"Saved {len(data)} samples to {output_file}")

if __name__ == "__main__":
    # Generate structured data based on a simple rule
    data = generate_data(num_samples=1000, input_size=10)
    
    # Save generated data to JSONL file
    save_to_jsonl(data, output_file='output/data.jsonl')
