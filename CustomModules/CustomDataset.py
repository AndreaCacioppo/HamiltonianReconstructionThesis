from torch.utils.data import Dataset, TensorDataset

#Create class for dataset
class custom_dataset_1d(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor # Encodes values of k
        self.y = y_tensor # Encodes bands (2-dim tensor)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.x.shape[0]
