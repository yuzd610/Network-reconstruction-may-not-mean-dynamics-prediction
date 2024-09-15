import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm

Disturb = 0.01
var_12 = np.sqrt(1 - Disturb ** 2)
delta_t = 0.01
step_t = 10000
N = 1000


n = 5
X_n_1 = np.zeros((N, step_t + 1))
X_n_2 = np.zeros((N, step_t + 1))

X_n_1[:, 0] = np.random.normal(0, 1, N)
X_n_2[:, 0] = X_n_1[:, 0]


LLE = []
G = []


for g in tqdm(np.arange(0.0001, 5.0001, 0.02)):
    r = np.zeros(n)

    for i in range(5):
        J_1 = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))
        J_11 = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))

        np.fill_diagonal(J_1, 0)
        np.fill_diagonal(J_11, 0)
        J_2 = var_12 * J_1 + J_11 * Disturb

        for t in range(0, step_t):
            X_n_1[:, t + 1] = (1 - delta_t) * X_n_1[:, t] + delta_t * np.matmul(J_1, np.tanh(X_n_1[:, t]))

        X_n_2[:, 1] = (1 - delta_t) * X_n_2[:, 0] + delta_t * np.matmul(J_2, np.tanh(X_n_2[:, 0]))
        ε = np.sqrt(np.sum((X_n_2[:, 1] - X_n_1[:, 1]) ** 2))
        lle = []

        for t in range(1, step_t):
            X_n_2[:, t + 1] = (1 - delta_t) * X_n_2[:, t] + delta_t * np.matmul(J_2, np.tanh(X_n_2[:, t]))

            d = np.sqrt(np.sum((X_n_2[:, t + 1] - X_n_1[:, t + 1]) ** 2))

            λ = np.log(d / ε) / delta_t
            X_n_2[:, t + 1] = X_n_1[:, t + 1] + ε * (X_n_2[:, t + 1] - X_n_1[:, t + 1]) / (
                np.linalg.norm((X_n_2[:, t + 1] - X_n_1[:, t + 1])))
            lle.append(λ)

        lle = lle[3000:]
        r[i] = np.mean(lle)
    LLE.append(r)
    G.append(g)



print(LLE)


mean_LLE = np.mean(LLE, axis=1)  # 对 n 轴求均值，得到 (step_t,)
std_LLE = np.std(LLE, axis=1)    # 对 n 轴求标准差，得到 (step_t,)

# 绘图








plt.plot(G, mean_LLE, linestyle='dashed', color='blue', linewidth=3)  # 设置为蓝色虚线
plt.fill_between(G, mean_LLE - std_LLE, mean_LLE + std_LLE, color='blue', alpha=0.2)

plt.xlabel('g', fontsize=40)
plt.ylabel('λ', fontsize=40)

# 设置刻度字体大小
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# 调整刻度线的大小和样式
plt.tick_params(axis='both', which='major', labelsize=35, length=10, width=2)

# 控制刻度间隔，确保刻度标签不重叠
plt.locator_params(axis='x', nbins=4)  # 控制x轴的刻度数量（可根据需要调整）
plt.locator_params(axis='y', nbins=4)  # 控制y轴的刻度数量（可根据需要调整）

# 自动调整布局
plt.tight_layout()

# 保存为PDF文件
plt.savefig('PLOT.pdf')

# 显示图形
plt.show()



















