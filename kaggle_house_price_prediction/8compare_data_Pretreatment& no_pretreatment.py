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

# 设置默认张量类型为浮点型
torch.set_default_tensor_type(torch.FloatTensor)

# 读取数据
train_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\train.csv')
test_data = pd.read_csv('D:\\Python\\kaggle_house_price_prediction\\house-prices-advanced-regression-techniques\\test.csv')

# 原始数据（不做任何预处理）
raw_train_features = train_data.iloc[:, 1:-1]
raw_test_features = test_data.iloc[:, 1:]

# 检查并转换所有列的数据类型为数值类型
raw_train_features = raw_train_features.apply(pd.to_numeric, errors='coerce')  # 将非数字数据转为NaN
raw_test_features = raw_test_features.apply(pd.to_numeric, errors='coerce')

# 填充缺失值，使用0或均值进行填充
raw_train_features = raw_train_features.fillna(0)
raw_test_features = raw_test_features.fillna(0)

# 数据预处理（标准化、缺失值填充和虚拟变量处理）
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
all_features = all_features.fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

# 数据转换为Tensor
n_train = train_data.shape[0]
# 做数据预处理的版本
train_features_processed = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features_processed = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 原始数据的版本
train_features_raw = torch.tensor(raw_train_features.values, dtype=torch.float)
test_features_raw = torch.tensor(raw_test_features.values, dtype=torch.float)

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
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, xlabel2=None, ylabel2=None, xscale='linear'):
    plt.figure(figsize=(8, 4))
    
    # 绘制第一条曲线，蓝色
    plt.plot(x_vals, y_vals, label='Processed')  # 给蓝色曲线添加label
    
    if x2_vals and y2_vals:
        plt.twinx()  # 创建第二个y轴
        plt.plot(x2_vals, y2_vals, 'r', label='Raw')  # 给红色曲线添加label
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    
    if legend:
        plt.legend()  # 自动显示图例
    if xlabel2 and ylabel2:
        plt.ylabel(ylabel2)
    
    plt.grid(True)
    plt.show()



# 对比实验：训练和预测
def train_and_pred(train_features_processed, test_features_processed, 
                   train_features_raw, test_features_raw, train_labels, test_data, 
                   num_epochs, lr, weight_decay, batch_size):
    net_processed = get_net(train_features_processed.shape[1])
    net_raw = get_net(train_features_raw.shape[1])

    # 训练数据预处理后的模型
    train_ls_processed, test_ls_processed = train(net_processed, train_features_processed, train_labels, 
                                                  test_features_processed, None, num_epochs, lr, weight_decay, batch_size)

    # 训练原始数据模型
    train_ls_raw, test_ls_raw = train(net_raw, train_features_raw, train_labels, 
                                      test_features_raw, None, num_epochs, lr, weight_decay, batch_size)
    
    # 绘制对比的RMSE曲线
    semilogy(range(1, num_epochs + 1), train_ls_processed, 'epochs', 'rmse', 
             range(1, num_epochs + 1), train_ls_raw, legend=['Processed', 'Raw'], xscale='linear')

    print(f"Processed train rmse: {train_ls_processed[-1]}")
    print(f"Raw train rmse: {train_ls_raw[-1]}")
    
    # 生成预测值
    preds_processed = net_processed(test_features_processed).detach().numpy()
    preds_raw = net_raw(test_features_raw).detach().numpy()

    # 将预测值保存到submission文件
    test_data['SalePrice_processed'] = pd.Series(preds_processed.reshape(1, -1)[0])
    test_data['SalePrice_raw'] = pd.Series(preds_raw.reshape(1, -1)[0])
    
    submission = pd.concat([test_data['Id'], test_data[['SalePrice_processed', 'SalePrice_raw']]], axis=1)
    submission.to_csv('D:\\Python\\kaggle_house_price_prediction\\output\\submission.csv', index=False)
    print("Submission saved successfully.")

# 调用训练和预测函数
train_and_pred(train_features_processed, test_features_processed, 
               train_features_raw, test_features_raw, train_labels, test_data, 
               num_epochs, lr, weight_decay, batch_size)
