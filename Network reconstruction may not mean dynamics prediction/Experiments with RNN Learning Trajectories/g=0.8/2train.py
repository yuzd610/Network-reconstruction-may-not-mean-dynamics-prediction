from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error ,num_epochs ,batch_size ,Truncate,set_num_sect
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20  # 可以统一设置字体大小
J = np.load('J_cupy.npy')

X = np.load("X.npy")

X = torch.from_numpy(X[:set_num_sect, :, :Truncate])

X = torch.chunk(X, chunk, dim=2)

X = torch.cat(X, dim=0)



class DynamicModel(nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.weight = nn.Parameter(torch.rand(N, N, dtype=torch.float64))


    def forward(self, trajectory):

        trajectory_point = trajectory
        reconstruct_trajectory = trajectory.clone()



        for p in range(reconstruct_trajectory.shape[2] - 1):
            current_points = trajectory_point[:, :, p]  
            next_points = (1 - delta_t) * current_points + delta_t * torch.matmul(self.weight, torch.tanh(trajectory_point[:, :, p].t())).t()

            reconstruct_trajectory[:, :, p + 1] = next_points

        return reconstruct_trajectory




model = DynamicModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)

losses = []
weight_losses = []

train_dataset = TensorDataset(X, X)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)




for j in tqdm(range(num_epochs)):
    losss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        loss = mean_squared_error(outputs, labels)
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        losss = losss + loss.item()

    losses.append(losss)
    weight_losses.append(np.mean((model.weight.data.clone().numpy() - J) ** 2))






np.save('J_study.npy', model.weight.data.clone().numpy())


print(sorted(weight_losses)[:10])


fig, ax1 = plt.subplots()

# 绘制第一个向量在左侧坐标轴
ax1.plot(losses, color='r', label='Training Error', linewidth=3)
ax1.set_xlabel('Epoch', fontsize=23)  # 增加字体大小
ax1.set_ylabel('Training Error', color='r', fontsize=23)  # 增加字体大小
ax1.tick_params(axis='y', labelcolor='r', labelsize=21)  # 增加刻度标签大小
ax1.tick_params(axis='x', labelsize=21)
# 创建右侧坐标轴
ax2 = ax1.twinx()

# 绘制第二个向量在右侧坐标轴
ax2.plot(weight_losses, color='g', label='Weight Distance', linewidth=3)
ax2.set_ylabel('Weight Distance', color='g', fontsize=23)  # 增加字体大小
ax2.tick_params(axis='y', labelcolor='g', labelsize=21)  # 增加刻度标签大小

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)  # 增加图例字体大小
plt.tight_layout()
fig.savefig('plot.pdf')
# 显示图形
plt.show()




losses = losses[-3:]

weight_losses = weight_losses[-3:]



fig, ax1 = plt.subplots()

# 绘制第一个向量在左侧坐标轴
ax1.plot(losses, color='b', label='Losses')
ax1.set_xlabel('epoch')
ax1.set_ylabel('Losses', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 创建右侧坐标轴
ax2 = ax1.twinx()

# 绘制第二个向量在右侧坐标轴
ax2.plot(weight_losses, color='r', label='Weight Losses')
ax2.set_ylabel('Weight Losses', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')
plt.tight_layout()

# 显示图形

plt.show()












