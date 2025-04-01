from torch.utils.data import Dataset

class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transforms):
        super().__init__()
        self.base_dataset = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]

        data = self.transforms(data)

        return data
