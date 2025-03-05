from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error ,num_epochs ,batch_size ,Truncate,set_num_sect
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
N1 = N

step_t = 30000
T = step_t*delta_t
J_1 = np.load('J_cupy.npy')
J_2 = np.load('J_study.npy')

X_n_1 = np.zeros((N1, step_t + 1))
X_n_2 = np.zeros((N1, step_t + 1))


X_n_1[:, 0] =  np.random.normal(0, 1, N1)
X_n_2[:, 0] =  X_n_1[:, 0]



for t in tqdm(range(0, step_t)):
    # Compute the update of X_n_1
    k1_1 = -X_n_1[:, t] + np.matmul(J_1, np.tanh(X_n_1[:, t]))
    k2_1 = - (X_n_1[:, t] + 0.5 * delta_t * k1_1) + np.matmul(J_1, np.tanh(X_n_1[:, t] + 0.5 * delta_t * k1_1))
    k3_1 = - (X_n_1[:, t] + 0.5 * delta_t * k2_1) + np.matmul(J_1, np.tanh(X_n_1[:, t] + 0.5 * delta_t * k2_1))
    k4_1 = - (X_n_1[:, t] + delta_t * k3_1) + np.matmul(J_1, np.tanh(X_n_1[:, t] + delta_t * k3_1))

    X_n_1[:, t + 1] = X_n_1[:, t] + (delta_t / 6) * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)

    #Compute the update for X_n_2
    k1_2 = -X_n_2[:, t] + np.matmul(J_2, np.tanh(X_n_2[:, t]))
    k2_2 = - (X_n_2[:, t] + 0.5 * delta_t * k1_2) + np.matmul(J_2, np.tanh(X_n_2[:, t] + 0.5 * delta_t * k1_2))
    k3_2 = - (X_n_2[:, t] + 0.5 * delta_t * k2_2) + np.matmul(J_2, np.tanh(X_n_2[:, t] + 0.5 * delta_t * k2_2))
    k4_2 = - (X_n_2[:, t] + delta_t * k3_2) + np.matmul(J_2, np.tanh(X_n_2[:, t] + delta_t * k3_2))

    X_n_2[:, t + 1] = X_n_2[:, t] + (delta_t / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)


time = np.linspace(0, T, step_t+1)

colors = ['red', 'green', 'blue']
fig, ax = plt.subplots()

# Draw some row data
for i in range(3):
    ax.plot(time, X_n_1[i, :], color=colors[i], linewidth=2, linestyle='-')  # 实线
    ax.plot(time, X_n_2[i, :], color=colors[i], linewidth=2, linestyle='--')  # 虚

# Set axis labels and titles
ax.set_xlabel('Time', fontsize=23)
ax.set_ylabel('Activity', fontsize=23)

# Set the tick label size
ax.tick_params(axis='both', which='major', labelsize=22)

# Set the legend font size
plt.tight_layout()
fig.savefig('plot.pdf')

plt.show()

