import torch
from torch.utils.data import Dataset, DataLoader
import json

class JSONLDataset(Dataset):
    def __init__(self, file_path):
        # set the data as empty
        self.data = []

        # open the file
        with open(file_path, 'r') as f:
            # loop through each line
            for line in f:
                # load the entry as json
                entry = json.loads(line)

                # set the features
                features = torch.tensor(entry['features'], dtype=torch.float32)

                # set the label
                label = torch.tensor(entry['label'], dtype=torch.long)

                # append
                self.data.append((features, label))

    def __len__(self):
        # get the length
        return len(self.data)

    def __getitem__(self, idx):
        # get the item
        return self.data[idx]

if __name__ == "__main__":
    # load the dataset
    dataset = JSONLDataset('output/data.jsonl')

    # load into a data loader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Iterate through the DataLoader and print batch information
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Features:", features)
        print("Labels:", labels)
        print()
