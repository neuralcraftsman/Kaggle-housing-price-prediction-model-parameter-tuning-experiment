import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 定义训练参数
num_epochs = 100  # 训练轮数
lr = 0.01         # 学习率
weight_decay = 0.001  # 权重衰减
batch_size = 64   # 批量大小

# 设置默认张量类型为浮点型
torch.set_default_tensor_type(torch.FloatTensor)

# 读取数据
train_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\train.csv')
test_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\test.csv')

# 预处理数据
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

# 数据格式转换
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 定义损失函数
loss = torch.nn.MSELoss()

# 定义模型
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

# 计算对数均方根误差（Log RMSE）
def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

# 训练模型
def train(net, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls = []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
    return train_ls

# K折交叉验证
def k_fold_cross_validation(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_train_rmse = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        print(f"Fold {fold + 1}/{k}")
        
        # 划分训练集和验证集
        train_fold_features, val_fold_features = train_features[train_idx], train_features[val_idx]
        train_fold_labels, val_fold_labels = train_labels[train_idx], train_labels[val_idx]
        
        # 创建并训练模型
        net = get_net(train_fold_features.shape[1])
        train_rmse = train(net, train_fold_features, train_fold_labels, num_epochs, lr, weight_decay, batch_size)
        
        # 计算验证集的RMSE
        val_rmse = log_rmse(net, val_fold_features, val_fold_labels)
        
        all_train_rmse.extend(train_rmse)
        print(f"Fold {fold + 1} training RMSE: {train_rmse[-1]:.4f}, validation RMSE: {val_rmse:.4f}")
    
    # 返回所有折的平均训练RMSE
    return np.mean(all_train_rmse)

# 获取不同k值下的RMSE
def evaluate_different_k_values(train_features, train_labels, num_epochs, lr, weight_decay, batch_size, max_k=10):
    k_values = range(2, max_k + 1)  # K 从 1 到 max_k
    avg_rmse_values = []
    
    for k in k_values:
        avg_rmse = k_fold_cross_validation(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
        avg_rmse_values.append(avg_rmse)
        print(f"K={k}, Average Training RMSE: {avg_rmse:.4f}")
    
    # 绘制K值与RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_rmse_values, marker='o')
    plt.xlabel("K (Number of Folds)")
    plt.ylabel("Average Training RMSE")
    plt.title("RMSE vs K (Number of Folds)")
    plt.grid(True)
    plt.show()

# 进行K值的评估
evaluate_different_k_values(train_features, train_labels, num_epochs, lr, weight_decay, batch_size)