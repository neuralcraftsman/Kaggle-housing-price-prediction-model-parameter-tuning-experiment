import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 定义训练参数
num_epochs = 100  # 训练轮数
lr = 11         # 学习率
weight_decay = 0.001  # 权重衰减
batch_size = 64   # 批量大小

torch.set_default_tensor_type(torch.FloatTensor)

# 读取数据
train_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\train.csv')
test_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\test.csv')

# 预处理数据
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features = all_features.fillna(0)

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

print(all_features.shape)

# 将数据格式转换成tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 定义损失函数
mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

# 绘图
def semilogy(x_vals, y_vals_mse, y_vals_mae, x_label, y_label, legend=None, xscale='linear'):
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals_mse, label='MSE')
    plt.plot(x_vals, y_vals_mae, label='MAE')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    if legend:
        plt.legend(legend)
    plt.grid(True)
    # 保存图表为PNG文件，确保在没有GUI环境下也能保存图表
    #plt.savefig('D:\\Python\\kaggle_house_price_prediction\\output\\training_loss_comparison.png')
    print("Training loss comparison saved as 'training_loss_comparison.png'.")
    plt.show()  # 显示图表

# 定义模型
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net    

# 计算R平方
def log_rmse(net, features, labels, loss_fn):
    with torch.no_grad():
        preds = net(features)
        clipped_preds = torch.clamp(preds, min=1.0)
        rmse = torch.sqrt(loss_fn(clipped_preds.log(), labels.log()))
    return rmse.item()

# 训练模型
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, loss_fn):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)  # 创建数据集
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)  # 创建数据加载器
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 定义优化器
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss_fn(net(X.float()), y.float())  # 使用传入的损失函数计算损失
            optimizer.zero_grad()  # 清零梯度
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
        train_ls.append(log_rmse(net, train_features, train_labels, loss_fn))  # 记录训练集上的RMSE
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels, loss_fn))  # 记录测试集上的RMSE
    return train_ls, test_ls

# 预测并保存结果
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size, loss_fn, loss_name):
    net = get_net(train_features.shape[1])  # 获取模型
    
    # 训练模型
    print(f"Training with {loss_name} Loss:")
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size, loss_fn)
    print(f'train rmse ({loss_name}) %f' % train_ls[-1])
    
    # 生成预测值
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.flatten())
    
    # 保存结果到不同文件
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission_file = f'D:\\Python\\kaggle_house_price_prediction\\output\\submission_{loss_name.lower()}.csv'
    print(f"Saving submission to {submission_file}...")
    submission.to_csv(submission_file, index=False)
    print(f"Submission saved successfully to {submission_file}.")

    # 返回训练损失，用于绘图
    return train_ls

# 分别使用 MSE 和 MAE 进行训练并保存结果
train_ls_mse = train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size, mse_loss, 'MSE')
train_ls_mae = train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size, mae_loss, 'MAE')

# 绘图，比较 MSE 和 MAE 的训练过程
semilogy(range(1, num_epochs + 1), train_ls_mse, train_ls_mae, 'epochs', 'rmse', legend=['MSE', 'MAE'])
