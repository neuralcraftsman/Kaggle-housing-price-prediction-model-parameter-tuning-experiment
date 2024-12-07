import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 定义训练参数
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

# 定义模型
def get_net(feature_num):
    # 创建一个线性回归模型
    net = nn.Linear(feature_num, 1)
    # 初始化模型参数
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net    

# 计算对数均方根误差（Log RMSE）
def log_rmse(net, features, labels, epoch):
    with torch.no_grad():
        # 预测值取最大值以避免对数运算中的负数
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        # 计算对数均方根误差
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
        print('epoch %d, train rmse: %f' % (epoch, rmse))
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
        train_ls.append(log_rmse(net, train_features, train_labels, epoch))
        if test_labels is not None:
            # 记录测试集上的RMSE
            test_ls.append(log_rmse(net, test_features, test_labels, epoch))
    return train_ls, test_ls

# 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, xlabel2=None, ylabel2=None, xscale='linear'):
    plt.plot(x_vals, y_vals, label=legend[0] if legend else None)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, label=legend[1] if legend else None)

# 预测并保存结果
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    # 训练模型并返回训练误差
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # 返回训练误差
    return train_ls

# 比较不同训练轮数下的训练误差
epoch_values = [1, 10, 20, 50, 100, 200, 500, 1000, 2000]
plt.figure(figsize=(8, 6))

for num_epochs in epoch_values:
    print(f"Training with {num_epochs} epochs...")
    train_ls = train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'RMSE', legend=[f'{num_epochs} epochs'])

# 添加图例和标签
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend(title="Training Epochs")
plt.grid(True)
plt.show()