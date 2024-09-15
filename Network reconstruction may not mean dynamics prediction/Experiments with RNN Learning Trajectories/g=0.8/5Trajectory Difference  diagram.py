from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error ,num_epochs ,batch_size ,Truncate,set_num_sect
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20  # 可以统一设置字体大小
N1 = N

step_t = 2000
T = step_t*delta_t
J_1 = np.load('J_cupy.npy')
J_2 = np.load('J_study.npy')

X_n_1 = np.zeros((N1, step_t + 1))
X_n_2 = np.zeros((N1, step_t + 1))


X_n_1[:, 0] =  np.random.normal(0, 1, N1)
X_n_2[:, 0] =  X_n_1[:, 0]



for t in tqdm(range(0, step_t)):
    X_n_1[:, t + 1] = (1 - delta_t) * X_n_1[:, t] + delta_t * np.matmul(J_1, np.tanh(X_n_1[:, t]))
    X_n_2[:, t + 1] = (1 - delta_t) * X_n_2[:, t] + delta_t * np.matmul(J_2, np.tanh(X_n_2[:, t]))



time = np.linspace(0, T, step_t+1)

colors = ['red', 'green', 'blue']
fig, ax = plt.subplots()

# 绘制每行数据
for i in range(3):
    ax.plot(time, X_n_1[i, :], color=colors[i], linewidth=2, linestyle='-')  # 实线
    ax.plot(time, X_n_2[i, :], color=colors[i], linewidth=2, linestyle='--')  # 虚

# 设置坐标轴标签和标题
ax.set_xlabel('Time', fontsize=23)  # 增加字体大小
ax.set_ylabel('Activity', fontsize=23)  # 增加字体大小

# 设置刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=22)  # 刻度标签大小

# 设置图例字体大小
plt.tight_layout()
fig.savefig('plot.pdf')
# 显示图形
plt.show()

