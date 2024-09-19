
from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error ,num_epochs ,batch_size ,Truncate,set_num_sect
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20  # 可以统一设置字体大小
J = np.load('J_cupy.npy')
J_study = np.load('J_study.npy')


X =[]





for i in range(200):
    for j in range(200):
        X.append(np.array([J[i][j], J_study[i][j]]))

X = np.array(X)



distances = (X[:,1]-X[:,0])**2

indices = np.argsort(distances)[-40000:]

# 提取这些点


X= X[indices]





# 创建图形和轴
fig, ax = plt.subplots()

# 绘制散点图，通过颜色和大小增加视觉效果
sc = ax.scatter(X[:, 0], X[:, 1], linewidth=0.8)

# 添加颜色条，表示数据点大小的意义


# 设置轴标签
ax.set_xlabel("$J_{ij}^1$", fontsize=22)
ax.set_ylabel("$J_{ij}^2$", fontsize=22)

# 设置刻度的字体大小
ax.tick_params(axis='both', labelsize=20)

# 使用tight_layout自动调整布局
plt.tight_layout()

# 保存图形
plt.savefig('J.pdf')

# 显示图形
plt.show()