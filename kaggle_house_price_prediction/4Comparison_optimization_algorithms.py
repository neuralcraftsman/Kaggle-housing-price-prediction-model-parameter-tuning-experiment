import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 定义训练参数
num_epochs = 100  # 训练轮数
lr = 11           # 学习率
weight_decay = 0.001  # 权重衰减
batch_size = 64   # 批量大小

# 设置默认张量类型为浮点型
torch.set_default_tensor_type(torch.FloatTensor)

# 读取数据
train_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\train.csv')
test_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\test.csv')

# 预处理数据
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
all_features = all_features.fillna(0)

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
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
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, optimizer_type='adam'):
    train_ls, test_ls = [], []
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    # 创建数据加载器
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    # 选择优化器
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        # 使用更小的学习率（如0.01）来避免NaN问题
        optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01, weight_decay=weight_decay)
    
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
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)  # 裁剪最大梯度为5
            
            # 更新参数
            optimizer.step()
        
        # 记录训练集上的RMSE
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            # 记录测试集上的RMSE
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label, legend=None, xscale='linear'):
    plt.figure(figsize=(8, 4))
    for label, y in y_vals.items():
        plt.plot(x_vals, y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    if legend:
        plt.legend(legend)
    plt.grid(True)
    plt.show()

# 预测并保存结果
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    # 训练模型并记录每种优化器下的训练损失
    adam_train_ls, _ = train(net, train_features, train_labels, None, None,
                             num_epochs, lr, weight_decay, batch_size, optimizer_type='adam')
    
    # 重置网络并训练SGD优化器
    net = get_net(train_features.shape[1])
    sgd_train_ls, _ = train(net, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size, optimizer_type='sgd')
    
    # 绘制训练过程中的RMSE曲线
    semilogy(range(1, num_epochs + 1), 
             {'Adam': adam_train_ls, 'SGD': sgd_train_ls}, 
             'epochs', 'train RMSE', 
             legend=['Adam', 'SGD'])
    
    print(f"Adam final train RMSE: {adam_train_ls[-1]}")
    print(f"SGD final train RMSE: {sgd_train_ls[-1]}")
    
    # 生成预测值
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.flatten())
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    print(submission.head())
    print("Saving submission to CSV...")
    submission.to_csv('D:\\Python\\kaggle_house_price_prediction\\output\\submission.csv', index=False)
    print("Submission saved successfully.")

# 调用训练和预测函数
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
