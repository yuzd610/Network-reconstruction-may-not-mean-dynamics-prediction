import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import iterations, g, set_num
import matplotlib.ticker as ticker
J_1 = np.load('J_cupy.npy')
J_2 = np.load('J_study.npy')


L = np.sum((J_2- J_1) ** 2)

print(L)



delta_t = 0.01
step_t = 1000
step_c = 30
N = 20000
N1 = 1000

n = 30
T = step_t * delta_t
Disturb = np.sqrt(1-(1-L/(2*(N1-1)*g**2))**2)
var_12 = np.sqrt(1-Disturb**2)
X_1 = np.zeros((N, step_t + 1))
X_2 = np.zeros((N, step_t + 1))
eta = np.zeros((N, 2*step_t, step_c + 1))
C_11 = np.zeros((step_t, step_c))
C_12 = np.zeros((step_t, step_c))
C_22 = np.zeros((step_t, step_c))
x_0 = np.random.normal(0, 1, N)
print(Disturb)
###初始化
mean = np.zeros(2*step_t)
covariance = np.eye(2*step_t)
eta[:, :, 0] = np.random.multivariate_normal(mean, covariance,N)
X_1[:, 0] = x_0
X_2[:, 0] = x_0



for i in tqdm(range(step_c)):
    for t in range(step_t):
        X_1[:, t + 1] = X_1[:, t] - delta_t * X_1[:, t] + delta_t * eta[:, t, i]
        X_2[:, t + 1] = X_2[:, t] - delta_t * X_2[:, t] + delta_t * eta[:, step_t+t, i]
        C_11[t, i] = np.mean(X_1[:, t] ** 2)
        C_12[t, i] = np.mean(X_2[:, t] *X_1[:, t])
        C_22[t, i] = np.mean(X_2[:, t] ** 2)


    tanh_X_1 = np.tanh(X_1[:, :-1])
    tanh_X_2 = np.tanh(X_2[:, :-1])
    product_matrix11 = np.dot(tanh_X_1.T, tanh_X_1) / X_1.shape[0]
    product_matrix12 = np.dot(tanh_X_1.T, tanh_X_2) / X_2.shape[0]
    product_matrix21 = product_matrix12.T
    product_matrix22 = np.dot(tanh_X_2.T, tanh_X_2) / X_2.shape[0]
    covariance[:step_t, :step_t] =  g ** 2 *product_matrix11
    covariance[:step_t, step_t:2 * step_t] = var_12 * g ** 2 *product_matrix12
    covariance[step_t:2 * step_t, :step_t] = var_12 * g ** 2 *product_matrix21
    covariance[step_t:2 * step_t, step_t:2 * step_t] =  g ** 2 *product_matrix22
    eta[:,:, i + 1] = np.random.multivariate_normal(mean, covariance,N)


d = C_22[:, -1] + C_11[:, -1] - 2*C_12[:, -1]








X_n_1 = np.zeros((N1, step_t + 1))
X_n_2 = np.zeros((N1, step_t + 1))


X_n_1[:, 0] =  np.random.normal(0, 1, N1)
X_n_2[:, 0] =  X_n_1[:, 0]

d_n = np.zeros((n, step_t))


for i in tqdm(range(n)):
    X_n_1[:, 0] = np.random.normal(0, 1, N1)
    X_n_2[:, 0] = X_n_1[:, 0]

    for t in tqdm(range(0, step_t)):
        X_n_1[:, t + 1] = (1 - delta_t) * X_n_1[:, t] + delta_t * np.matmul(J_1, np.tanh(X_n_1[:, t]))
        X_n_2[:, t + 1] = (1 - delta_t) * X_n_2[:, t] + delta_t * np.matmul(J_2, np.tanh(X_n_2[:, t]))

        c = np.mean((X_n_1[:, t] - X_n_2[:, t]) ** 2)
        d_n[i, t] = c


time = np.linspace(0, T, step_t)

mean_d_n = np.mean(d_n, axis=0)  # 对 n 轴求均值，得到 (step_t,)
std_d_n = np.std(d_n, axis=0)    # 对 n 轴求标准差，得到 (step_t,)

# 绘图
plt.rcParams['axes.formatter.limits'] = (-3, 3)
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.labelsize'] = 23
plt.rcParams['ytick.labelsize'] = 23







plt.plot(time, d, label=r'DMFT', linestyle='solid', color='red',linewidth=3)  # 将虚线改为实线并设置为红色
plt.plot(time, mean_d_n, label=r'simulation', linestyle='dashed', color='blue',linewidth=3)  # 设置为蓝色虚线
plt.fill_between(time, mean_d_n - std_d_n, mean_d_n + std_d_n, color='blue', alpha=0.2)
# 设置坐标轴标签，并指定字体大小
plt.xlabel('Time ', fontsize=23)
plt.ylabel('Distance', fontsize=23)

# 设置刻度字体大小
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

# 添加图例，并指定字体大小
plt.legend(fontsize=18)

# 启用网格

plt.tight_layout()
plt.savefig('plot.pdf')
# 显示图形
plt.show()

