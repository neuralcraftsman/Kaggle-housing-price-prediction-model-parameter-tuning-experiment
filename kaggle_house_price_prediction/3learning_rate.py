import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 定义训练参数
num_epochs = 100  # 训练轮数
lr_list = range(1, 201, 10)  # 学习率从1到200，步长10
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
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
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
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
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
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label, legend=None, xscale='linear'):
    plt.figure(figsize=(8, 4))
    for lr, y in y_vals.items():
        plt.plot(x_vals, y, label=f"lr={lr}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    if legend:
        plt.legend(legend)
    plt.grid(True)
    plt.show()

# 预测并保存结果
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr_list, weight_decay, batch_size):
    all_train_ls = {}  # 用于存储每个学习率下的训练损失
    for lr in lr_list:
        net = get_net(train_features.shape[1])
        train_ls, _ = train(net, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size)
        all_train_ls[lr] = train_ls
        print(f"Learning Rate {lr}: final train RMSE = {train_ls[-1]}")

        # 生成预测值
        preds = net(test_features).detach().numpy()
        test_data['SalePrice'] = pd.Series(preds.flatten())
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission_file = f'D:\\Python\\kaggle_house_price_prediction\\output\\submission_lr_{lr}.csv'
        submission.to_csv(submission_file, index=False)
        print(f"Submission for lr={lr} saved successfully.")
    
    # 绘制不同学习率下的训练RMSE曲线
    semilogy(range(1, num_epochs + 1), all_train_ls, 'epochs', 'train rmse', legend=[f'lr={lr}' for lr in lr_list])

# 调用训练和预测函数
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr_list, weight_decay, batch_size)
