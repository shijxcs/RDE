import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from scipy.sparse import csr_matrix, coo_matrix

'''
class XMLDataset(Dataset):
    def __init__(self, features: csr_matrix, scores: csr_matrix, labels: csr_matrix = None, training=True):
        self.features, self.scores, self.labels, self.training = features, scores, labels, training
        
    def __getitem__(self, index):
        feature = self.features[index].toarray().squeeze(0).astype(np.float32)
        score = self.scores[index].toarray().squeeze(0).astype(np.float32)
        if self.training:
            label = self.labels[index].toarray().squeeze(0).astype(np.float32)
            return feature, score, label
        else:
            return feature, score
               
    def __len__(self):
        return self.features.shape[0]
'''

def csr2arr(a: csr_matrix, maxlen):
    col = np.pad(a.indices, (0, maxlen - len(a.indices)), 'constant', constant_values=0).astype('int64')
    value = np.pad(a.data, (0, maxlen - len(a.data)), 'constant', constant_values=0).astype('float32')
    return col, value
    

class XMLDataset(Dataset):
    def __init__(self, features: csr_matrix, scores: csr_matrix, labels: csr_matrix = None, training=True):
        self.features, self.scores, self.labels, self.training = features, scores, labels, training
        self.ft_maxlen = max(features.indptr[1:] - features.indptr[:-1])
        self.sc_maxlen = max(scores.indptr[1:] - scores.indptr[:-1])
        self.lbl_maxlen = max(labels.indptr[1:] - labels.indptr[:-1])
        #print(self.ft_maxlen, self.sc_maxlen, self.lbl_maxlen)
    
    def __getitem__(self, index):
        ft_col, ft_value = csr2arr(self.features[index], self.ft_maxlen)
        sc_col, sc_value = csr2arr(self.scores[index], self.sc_maxlen)
        if self.training:
            lbl_col, lbl_value = csr2arr(self.labels[index], self.lbl_maxlen)
            return ft_col, ft_value, sc_col, sc_value, lbl_col, lbl_value
        else:
            return ft_col, ft_value, sc_col, sc_value
        
    def __len__(self):
        return self.features.shape[0]
    
    
class NormedLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_size, out_size))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        
    def forward(self, x):
        out = x.mm(F.normalize(self.weight, dim=0))
        return out

    
class MLLinear(nn.Module):
    def __init__(self, feature_size, hidden_size, label_size,
                 input_mode='feature', output_mode='residual', use_norm=True, drop_prob=0.5, **kwargs):
        super(MLLinear, self).__init__()
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.use_norm = use_norm
        
        shape = self.get_shape(feature_size, hidden_size, label_size)
        self.fc_list, self.output_fc = self.get_linear(shape)
        self.bn_list = nn.ModuleList( nn.BatchNorm1d(out_s) for out_s in shape[1:-1] )
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, features, scores):
        features = F.normalize(features, dim=1)
        scores = F.normalize(scores, dim=1)
        
        out = self.get_input(features, scores)
        for fc, bn in zip(self.fc_list, self.bn_list):
            out = fc(out)
            out = bn(out)
            out = F.relu(out)
        out = self.dropout(out)
        out = self.output_fc(out)
        out = self.get_output(out, scores)
        return out
    
    def get_shape(self, feature_size, hidden_size, label_size):
        if self.input_mode == 'feature':
            return (feature_size,) + hidden_size + (label_size,)
        elif self.input_mode == 'score':
            return (label_size,) + hidden_size + (label_size,)
        elif self.input_mode == 'combined':
            return (feature_size + label_size,) + hidden_size + (label_size,)
    
    def get_input(self, ft, sc):
        if self.input_mode == 'feature':
            return ft
        elif self.input_mode == 'score':
            return sc
        elif self.input_mode == 'combined':
            return torch.cat((ft, sc), dim=1)
    
    def get_linear(self, shape):
        fc_list = nn.ModuleList( nn.Linear(in_s, out_s) for in_s, out_s in zip(shape[0:-2], shape[1:-1]) )
        for fc in fc_list:
            nn.init.xavier_uniform_(fc.weight)
        if self.use_norm:
            output_fc = NormedLinear(shape[-2], shape[-1])
        else:
            output_fc = nn.Linear(shape[-2], shape[-1])
            nn.init.xavier_uniform_(output_fc.weight)
        return fc_list, output_fc
    
    def get_output(self, out, sc):
        if self.output_mode == 'default':
            return out
        elif self.output_mode == 'residual':
            sc = sc.clamp(min=1e-6, max=1-1e-6)
            out += torch.log(sc)
            out -= torch.log(1 - sc)
            return out
        