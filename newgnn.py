import numpy as np 
import pandas as pd

import os 
import json 
import argparse
import torch 
import gc 
from joblib import Parallel, delayed
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.io import read_txt_array
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
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super(LoRALinear, self).__init__()
        self.low_rank_a = nn.Linear(in_features, r, bias=False)
        self.low_rank_b = nn.Linear(r, out_features, bias=False)
        self.scaling = 1.0 / r

    def forward(self, x):
        return self.low_rank_b(self.low_rank_a(x)) * self.scaling
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,initial_embedding,num_nodes=3, embedding_dim=4096):
        super(GCN, self).__init__()
        
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.embedding.weight.data.copy_(initial_embedding)
        # 第一层图卷积
        self.conv1 = GCNConv(input_dim+4096, hidden_dim)
        self.lora1 = LoRALinear(hidden_dim, hidden_dim, r=8)
        # 第二层图卷积
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.lora2 = LoRALinear(output_dim, output_dim, r=8)

    def forward(self, x, edge_index,domain_idx):
        # 第一层卷积 + 激活函数
        embedded = self.embedding(domain_idx).view(domain_idx.size(0), -1)
        # concatenate with X
        x = torch.cat([embedded, x.float()], dim=1)
        
        x = self.conv1(x, edge_index)
        x = self.lora1(x)  # LoRA 作用
        x = F.relu(x)

        # 第二层卷积 + LoRA
        x = self.conv2(x, edge_index)
        x = self.lora2(x)
        return F.log_softmax(x, dim=1)  # 使用 LogSoftmax，适用于分类任务
 #主函数   
 #ACM  dataset
def loadxy(taskname):
    docs_path = "/public12_data/fl/Citation/"+taskname+"/raw/"+taskname+"_docs.txt"
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)
    x = torch.from_numpy(x).to(torch.float)

    label_path = "/public12_data/fl/Citation/"+taskname+"/raw/"+taskname+"_labels.txt"
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)
    y = torch.from_numpy(y).to(torch.int64)
    edge_path = "/public12_data/fl/Citation/"+taskname+"/raw/"+taskname+"_edgelist.txt"
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()
    return x,y,edge_index
def train(model, x,y,edge_index, optimizer):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除旧的梯度
    x = torch.tensor(x, dtype=torch.float32)
    x=x[0:256]
    x, domain_idx = x[:,:-1], x[:,-1]
    domain_idx = torch.tensor(domain_idx, dtype=torch.long)
    y=y[0:256]
    edge_index = edge_index[:, edge_index.max(dim=0).values < 256]
    out = model(x, edge_index,domain_idx)  # 前向传播
    loss = F.nll_loss(out, y)  # 计算损失（没有使用train_mask，直接使用所有数据）
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss.item()
def retrain(model, x,y,edge_index, optimizer):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除旧的梯度
    x = torch.tensor(x, dtype=torch.float32)
    x, domain_idx = x[:,:-1], x[:,-1]
    domain_idx = torch.tensor(domain_idx, dtype=torch.long)
    loss = F.nll_loss(out[0:128], y[0:128])  # 计算损失（没有使用train_mask，直接使用所有数据）
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss.item()
def test(model, x,y,edge_index):
    model.eval()  # 设置模型为评估模式
    x = torch.tensor(x, dtype=torch.float32)
    x, domain_idx = x[:,:-1], x[:,-1]
    domain_idx = torch.tensor(domain_idx, dtype=torch.long)
    out = model(x, edge_index,domain_idx)
    pred = out.argmax(dim=1)  # 获取每个节点的预测标签
    accuracy = (pred == y).sum().item() / y.size(0)  # 计算准确率
    return accuracy
initial_embedding_path = f"/public12_data/fl/embed/prompt.npy"
initial_embedding = torch.from_numpy(np.load(initial_embedding_path)).float()
print(initial_embedding.shape)


x,y,edge_index=loadxy("Citationv1")

# 将 initial_embedding 的第一行添加到 x 的每一行
zeros_column = np.zeros((x.shape[0], 1))


input_dim = x.size(1)  # 输入特征的维度
hidden_dim = 64  # 隐藏层维度
output_dim = 5  # 类别数
model = GCN(input_dim, hidden_dim, output_dim,initial_embedding)

    # 优化器
optimizer = Adam(model.parameters(), lr=0.01)

    # 训练

model.load_state_dict(torch.load("gcn_model_weights.pth"))

x,y,edge_index=loadxy("ACMv9")
ones_column = np.ones((x.shape[0], 1))

accuracy = test(model, np.hstack([x, ones_column]),y,edge_index)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
optimizer = torch.optim.Adam(model.embedding.parameters(), lr=0.1)
for epoch in range(50):
    loss = train(model, np.hstack([x, ones_column]),y,edge_index, optimizer)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # 测试
accuracy = test(model, np.hstack([x, ones_column]),y,edge_index)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
x,y,edge_index=loadxy("DBLPv7")
twos_column = 2 * np.ones((x.shape[0], 1))
accuracy = test(model, np.hstack([x, twos_column]),y,edge_index)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

ones_column = np.ones((x.shape[0], 1))
accuracy = test(model, np.hstack([x, ones_column]),y,edge_index)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
input_dim = x.shape[1]    # 输入特征维度
hidden_dim = 32   # 隐藏层维度
output_dim = 5    # 输出类别数
promodel=MLPe5Classifier(input_dim, num_classes = 38, hidden_dim=64,refit_epochs=1000)

initial_embedding_path = f"/public12_data/fl/embed/incontext64.npy"
initial_embedding = torch.from_numpy(np.load(initial_embedding_path)).float()
zeros_column = np.zeros((x.shape[0], 1))

promodel.load(idx=10)
#测试集
docs_path = "/public12_data/fl/Citation/Citationv1/raw/Citationv1_docs.txt"
f = open(docs_path, 'rb')
content_list = []
for line in f.readlines():
    line = str(line, encoding="utf-8")
    content_list.append(line.split(","))
x = np.array(content_list, dtype=float)
x = torch.from_numpy(x).to(torch.float)

label_path = "/public12_data/fl/Citation/Citationv1/raw/Citationv1_labels.txt"
f = open(label_path, 'rb')
content_list = []
for line in f.readlines():
    line = str(line, encoding="utf-8")
    line = line.replace("\r", "").replace("\n", "")
    content_list.append(line)
y = np.array(content_list, dtype=int)
y = torch.from_numpy(y).to(torch.int64)
promodel.refit_epochs = 1000
promodel.refit_lr = 0.001
promodel.refit(x[0:128], y[0:128])
train_acc, train_f1 = promodel.score(x[128:], y[128:])
print(train_acc)