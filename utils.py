import pickle
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import json

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mae_loss_(y_pred, y_true, score = None):
    score = score.softmax(dim = -1).unsqueeze(dim = -1)
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    if score is not None:
        loss = torch.einsum('bhc,bhl->bcl',(loss,score))
    return loss.mean()

class StandardScaler():
    def __init__(self, mean, std, factor = 1.25):
        self.mean = mean
        self.std = std
        self.factor = factor
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return ((data-0.5) * 2 * self.std* self.factor) + self.mean
    
class StandardScaler_TaGNN():
    def __init__(self, mean, std, dmean, dstd, factor = 1.125):
        self.mean = mean
        self.std = std
        self.factor = factor
        self.mean_ = dmean
        self.std_ = dstd
    def transform(self, data):
        return (data - self.mean_) / self.std_
    def inverse_transform_(self, data): # mean is min, std is max - min
        return (((data-0.5)* self.factor +0.5) * self.std) + self.mean
    
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return  adj_mx

class DataLoaderST_TaGNN(object):
    def __init__(self, num, file_name, trend_name, data_paths, device, horizon, window):
        self.P = window 
        self.h = horizon 
        fin = np.load(file_name)
        self.rawdat = fin['data']
        tin = np.load(trend_name)
        self.trend = tin['data']
        self.label = pd.read_excel('.../data/标签数据.xlsx', engine='openpyxl')
        self.adj_ = pd.read_excel('.../data/构图数据.xlsx', engine='openpyxl')
        self.adj = load_graph_data('.../data/XIHU_Adj_Noise.pkl')
        self.n = len(self.rawdat)
        self.m = self.rawdat[0].shape[0]
        with open('.../data/split_dataset.json', 'r') as json_file:
            self.dict = json.load(json_file)
        self.index = self.cross_validation_indices(data_paths)
        self.num = num
        self._split(self.index, self.num)
        self.device = device
        vlabels = torch.cat([self.train[1], self.valid[1]],dim = 0)
        self.scalar = StandardScaler_TaGNN(mean = vlabels.min(dim = 0)[0][:-1].unsqueeze(dim = 0).to(self.device), \
                                      std = (vlabels.max(dim = 0)[0] - vlabels.min(dim = 0)[0])[:-1].unsqueeze(dim = 0).to(self.device), \
                                        dmean = self.train[0][:, :-1, :, :5].flatten(0,2).mean(dim = 0), \
                                            dstd = self.train[0][:, :-1, :, :5].flatten(0,2).std(dim = 0))
        
    def cross_validation_indices(self, data_paths):
        all_train_indices = []
        all_val_indices = []
        all_test_indices = []
        for j in range(5):
            train_data = pd.concat([pd.read_excel(data_paths[i]).set_index("index") for i in self.dict['train'][j]])
            val_data = pd.concat([pd.read_excel(data_paths[i]).set_index("index") for i in self.dict['valid'][j]])
            test_data = pd.concat([pd.read_excel(data_paths[i]).set_index("index") for i in self.dict['test'][j]])
            train_indices = train_data.index.tolist()
            val_indices = val_data.index.tolist()
            test_indices = test_data.index.tolist()
            
            all_train_indices.append(train_indices)
            all_val_indices.append(val_indices)
            all_test_indices.append(test_indices)
        return {"train": all_train_indices, "val": all_val_indices, "test": all_test_indices}

    def _split(self, index, num):
        train_set = index['train'][num]
        valid_set = index['val'][num]
        test_set = index['test'][num]
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set) 
        X = torch.zeros((n, self.m, self.P, 6))
        Y = torch.zeros((n, 6))
        A = torch.zeros((n, 39, 39))
        T = torch.zeros((n, self.m-1, self.P, 5))

        for i in range(n):
            X[i, :, :, :] = torch.from_numpy(self.dat[idx_set[i]])
            Y[i, :] = torch.from_numpy(self.label.iloc[idx_set[i]].values)
            A[i, :38, :38] = torch.from_numpy(self.adj)
            T[i, :, :, :] = torch.from_numpy(self.trend[idx_set[i]])
            A[i, self.adj_.iloc[idx_set[i]][0], 38] = 1
            A[i, self.adj_.iloc[idx_set[i]][1], 38] = 1
            A[i, 38, self.adj_.iloc[idx_set[i]][0]] = 1
            A[i, 38, self.adj_.iloc[idx_set[i]][1]] = 1
        return [X, Y, A, T]

    def get_batches(self, inputs, targets, adj, trends, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            T = trends[excerpt]
            A = adj[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            T = T.to(self.device)
            A = A.to(self.device)
            yield Variable(X), Variable(Y), Variable(A), Variable(T)
            start_idx += batch_size
