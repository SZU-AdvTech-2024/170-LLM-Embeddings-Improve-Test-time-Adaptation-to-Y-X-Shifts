import numpy as np 
import pandas as pd

import os 
import json 
import argparse
import torch 
import gc 
from joblib import Parallel, delayed
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from src.mlp_concat import *
from src.mlp_e5 import *
from src.mlp import *
from utils import (
    get_raw_data, 
    sample_config, 
    fetch_model, 
    sample_data, 
    sample_val_data
)
from sklearn.model_selection import train_test_split
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        # 第一层图卷积
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二层图卷积
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # 第一层卷积 + 激活函数
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层卷积
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # 使用 LogSoftmax，适用于分类任务
 #主函数   
data = pd.read_pickle('/public12_data/fl/embed/arxiv.pkl')
path="/public12_data/fl/graph_1.pt"
X = data["embedding"].values.tolist()
X = np.array(X).astype(float)
print(X[:100])
y = data['label'].values.tolist()
y = np.array(y).reshape(-1)
y = y.astype(np.int64)
y_tensor = torch.tensor(y)
unique_labels = torch.unique(y_tensor)

# 2. 构建一个映射表
label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

# 3. 将原始标签映射到新的连续范围
y = torch.tensor([label_map[label.item()] for label in y])
y = torch.tensor(y).long()
data2 = pd.read_pickle('/public12_data/fl/embed/arxiv_test.pkl')
X_test = data2["embedding"].values.tolist()
X_test = np.array(X_test).astype(float)

y_test = data2['label'].values.tolist()
y_test = np.array(y_test).reshape(-1)
y_test = y_test.astype(np.int64)
y_test = torch.tensor([label_map[label.item()] for label in y_test])
y_test = torch.tensor(y_test).long()


input_dim = 4096    # 输入特征维度
hidden_dim = 32   # 隐藏层维度
output_dim = 39    # 输出类别数
promodel=MLPconcatClassifier(num_classes=38,initial_embedding_method = 'arxiv64',num_embeddings=2,training_method = 'nn',refit_method = 'nn')
initial_embedding_path = f"/public12_data/fl/embed/incontext64.npy"
initial_embedding = torch.from_numpy(np.load(initial_embedding_path)).float()
zeros_column = np.zeros((X.shape[0], 1))

promodel.load(idx=2)
ones_column = np.ones((X_test.shape[0], 1))
np.hstack([X_test, ones_column])
train_acc, train_f1 = promodel.score(np.hstack([X_test, ones_column]), y_test)

print(train_acc)
domainlabel_model = MLPe5Classifier(input_dim, num_classes = 38, hidden_dim=64,refit_epochs=1000)

domainlabel_model.load(idx=1)
train_acc, train_f1 = domainlabel_model.score(X_test, y_test)

print(train_acc)
domainlabel_model.refit(X_test[:128],y_test[:128])
train_acc, train_f1 = domainlabel_model.score(X_test, y_test)

print(train_acc)
