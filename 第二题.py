import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# 创建模型
model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.011)

filename = "train 2.csv"

data = np.genfromtxt(filename,
                     delimiter=",",
                     usecols=[0, 1, 2, 3]
                     )
print(data.dtype)
array = np.array(data)
target = np.genfromtxt(filename,
                       delimiter=',',
                       usecols=[4],
                       dtype=float)
target = np.array(target)
print(target.dtype)
target = target[:, np.newaxis]  # 使用 np.newaxis 在第二维度上增加一个维度


# 构建数据集和数据加载器
train_dataset = torch.from_numpy(array).float()  # 根据实际情况构造训练数据集（假设输入数据为随机数）
train_labels = torch.from_numpy(target).float()  # 根据实际情况构造训练标签（假设标签为随机数）


train_data = torch.utils.data.TensorDataset(train_dataset, train_labels)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
print(len(train_loader))
# 返回模型的参数
parameters = model.state_dict()
print(parameters)