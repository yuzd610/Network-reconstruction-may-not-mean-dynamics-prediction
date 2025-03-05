from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error, num_epochs, batch_size, Truncate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

Dis = []
T = [ ]




p = np.load('p.npy')
mean_dw = np.load('mean_dw.npy')
std_d_n = np.load('std_d_n.npy')

step_t = 10000

dw_up = mean_dw +std_d_n
dw_down = mean_dw -std_d_n


def eta(N, g, d_W):
    term = 1 - N * d_W / (2 * g**2)
    return np.sqrt(1 - term**2)


eta_up=eta(N, g, dw_up)
eta_down=eta(N, g, dw_down )
eta_mean =eta(N, g, mean_dw)

print(eta_mean)
delta = 1

n = 5
X_n_1 = np.zeros((N, step_t + 1))
X_n_2 = np.zeros((N, step_t + 1))

X_n_1[:, 0] = np.random.normal(0, 1, N)
X_n_2[:, 0] = X_n_1[:, 0]


for Disturb in tqdm(eta_mean):
    print(Disturb)
    var_12 = np.sqrt(1 - Disturb ** 2)
    r = np.zeros(n)

    for i in range(n):
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

    T_pd = (1/r)*np.log(delta/ε)

    T.append(T_pd)



#Add mean and standard deviation bars

mean_LLE = np.mean(T, axis=1)
std_LLE = np.std(T, axis=1)
# 绘图





plt.plot(p, mean_LLE, linestyle='dashed', color='blue', linewidth=3)  # 设置为蓝色虚线
plt.fill_between(p, mean_LLE - std_LLE, mean_LLE + std_LLE, color='blue', alpha=0.2)

plt.xlabel(r'$P$', fontsize=20)   # $\eta$ 用 LaTeX 渲染希腊字母
plt.ylabel(r'$T_{pd}$', fontsize=20) # $T_{pd}$ 渲染带下标的 T


# Set the scale font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Adjust the size and style of the tick marks
plt.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)


plt.locator_params(axis='x', nbins=4)  # Control the number of x-axis ticks (adjustable as needed)
plt.locator_params(axis='y', nbins=4)  # Control the number of y-axis ticks (adjustable as needed)
plt.title(rf"$N = {N},\ g = {g},\ \delta = {delta}$", pad=20,fontsize=20)


plt.tight_layout()


plt.savefig('plot1.pdf')


plt.show()

















