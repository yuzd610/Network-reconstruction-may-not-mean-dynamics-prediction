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



X = np.load("X.npy")

data = X[0, :, :]

# 获取时间点，即列索引乘以 0.01
time_points = np.arange(data.shape[1]) * 0.01



fig, ax = plt.subplots()

# 绘制每行数据
for i in range(8):
    ax.plot(time_points, data[i, :],linewidth=2)


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20  # 可以统一设置字体大小
# 设置坐标轴标签和标题
ax.set_xlabel('Time', fontsize=23)  # 增加字体大小
ax.set_ylabel('Activity', fontsize=23)  # 增加字体大小

# 设置刻度标签大小
# 设置刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=20)

# 设置图例，暂无图例数据，假设有需要


# 使用tight_layout自动调整布局
plt.tight_layout()

# 保存图形
fig.savefig('plot.pdf')

# 显示图形
plt.show()