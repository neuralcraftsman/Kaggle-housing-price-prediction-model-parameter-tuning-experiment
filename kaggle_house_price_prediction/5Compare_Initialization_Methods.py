import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
# 将训练数据和测试数据的特征部分拼接在一起
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化
# 找出数值特征列
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 对数值特征进行标准化处理
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features = all_features.fillna(0)

# 将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

# 打印预处理后的特征矩阵形状
print(all_features.shape)

# 将数据格式转换成tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 定义损失函数
loss = torch.nn.MSELoss()
print(loss)

# 定义模型
def get_net_normal_init(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

def get_net_xavier_init(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
    return net

def get_net_kaiming_init(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        if len(param.shape) > 1:
            nn.init.kaiming_uniform_(param)
    return net

def get_net_constant_init(feature_num, value=0.01):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.constant_(param, value)
    return net

# 计算对数均方根误差（Log RMSE）
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 预测值取最大值以避免对数运算中的负数
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        # 计算对数均方根误差
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

# 训练模型
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    # 创建数据加载器
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 定义优化器
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 设置模型为浮点型
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in data_iter:
            # 计算损失
            l = loss(net(X.float()), y.float())
            # 清零梯度
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
        # 记录训练集上的RMSE
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            # 记录测试集上的RMSE
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, x3_vals=None, y3_vals=None, x4_vals=None, y4_vals=None,
             legend=None, xlabel2=None, ylabel2=None, xscale='linear'):
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=legend[0])
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, label=legend[1])
    if x3_vals and y3_vals:
        plt.plot(x3_vals, y3_vals, label=legend[2])
    if x4_vals and y4_vals:
        plt.plot(x4_vals, y4_vals, label=legend[3])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    if legend:
        plt.legend()
    plt.grid(True)
    plt.show()

# 使用正态分布初始化训练模型
net_normal = get_net_normal_init(train_features.shape[1])
train_ls_normal, _ = train(net_normal, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size)
print('Normal Initialization Train RMSE: %f' % train_ls_normal[-1])

# 使用 Xavier 初始化训练模型
net_xavier = get_net_xavier_init(train_features.shape[1])
train_ls_xavier, _ = train(net_xavier, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size)
print('Xavier Initialization Train RMSE: %f' % train_ls_xavier[-1])

# 使用 Kaiming He 初始化训练模型
net_kaiming = get_net_kaiming_init(train_features.shape[1])
train_ls_kaiming, _ = train(net_kaiming, train_features, train_labels, None, None,
                             num_epochs, lr, weight_decay, batch_size)
print('Kaiming He Initialization Train RMSE: %f' % train_ls_kaiming[-1])

# 使用常数初始化训练模型
net_constant = get_net_constant_init(train_features.shape[1], value=0.01)
train_ls_constant, _ = train(net_constant, train_features, train_labels, None, None,
                              num_epochs, lr, weight_decay, batch_size)
print('Constant Initialization Train RMSE: %f' % train_ls_constant[-1])

# 打印训练结果以确保数据正确
print("Train RMSE for Normal Initialization:", train_ls_normal)
print("Train RMSE for Xavier Initialization:", train_ls_xavier)
print("Train RMSE for Kaiming He Initialization:", train_ls_kaiming)
print("Train RMSE for Constant Initialization:", train_ls_constant)

# 绘制四种初始化方法的训练 RMSE 曲线
semilogy(range(1, num_epochs + 1), train_ls_normal, 'epochs', 'rmse', 
         range(1, num_epochs + 1), train_ls_xavier, 
         range(1, num_epochs + 1), train_ls_kaiming, 
         range(1, num_epochs + 1), train_ls_constant, 
         legend=['Normal Init', 'Xavier Init', 'Kaiming He Init', 'Constant Init'])