import numpy as np
from config import iterations, delta_t, N, g, set_num
from tqdm import tqdm

# 初始化权重矩阵 J
J = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))

np.fill_diagonal(J, 0)

np.save('J_cupy.npy', J)

# 初始化所有样本集的状态矩阵
X = np.random.normal(0, 1, size=(set_num, N, iterations + 1))

# 计算更新项只需一次，然后在迭代中重用
J_scaled = delta_t * J

# 迭代更新每个时间步
for i in tqdm(range(iterations)):
    X[:, :, i + 1] = (1 - delta_t) * X[:, :, i] + np.matmul(J_scaled, np.tanh(X[:, :, i]).T).T


np.save('X.npy', X)
