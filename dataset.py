import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def normalize_dataset(x):
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x != 0:
        return (x-mean_x)/std_x
    else:
        return x - mean_x


#%%
class BeatDataset(Dataset):
    def __init__(self, rootdir, datasets, seed = 123, N=0):
        np.random.seed(seed)
        self.rootdir = rootdir
        
        labels_all = []
        for dataset in datasets:
            labels = pd.read_csv(f'{rootdir}/{dataset}/labels.csv', index_col=0)
            labels.index = labels['dataset'] + '_' + labels.index.astype(str)
            labels_all.append(labels)
        
        labels = pd.concat(labels_all, axis=0)
        if N > 0:
            labels = labels.sample(n=N)
        self.labels = labels
        
    def __len__(self):
        return(len(self.labels))
    
    def __getitem__(self, idx):
        sample = self.labels.iloc[idx, :]
        dataset = sample['dataset']
        mode = sample['part_type']
        filename = f'{sample.name}.npy'
        
        signal = np.load(f'{self.rootdir}/{dataset}/{mode}/data/{filename}')
        signal = normalize_dataset(signal)
        label = sample['label']
        
        signal_out = torch.tensor(signal).float()
        target = torch.tensor(label)
        
        return({'data': signal_out, 'target': target, 'sample': sample.name, 'info': sample.to_dict()})

    
