import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm





Dis = []
T = [ ]


g = 3
N = 200



delta_t = 0.01
step_t = 10000

delta = 10

n = 5
X_n_1 = np.zeros((N, step_t + 1))
X_n_2 = np.zeros((N, step_t + 1))

X_n_1[:, 0] = np.random.normal(0, 1, N)
X_n_2[:, 0] = X_n_1[:, 0]

for Disturb in tqdm(np.arange(0.01, 1, 0.1)):
    Dis .append(Disturb)
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




# Calculate mean and standard deviation bars
mean_LLE = np.mean(T, axis=1)
std_LLE = np.std(T, axis=1)
# Plotting



print(mean_LLE)


print(Dis)

plt.plot(Dis, mean_LLE, linestyle='dashed', color='blue', linewidth=3)  # 设置为蓝色虚线
plt.fill_between(Dis, mean_LLE - std_LLE, mean_LLE + std_LLE, color='blue', alpha=0.2)

plt.xlabel(r'$\eta$', fontsize=20)
plt.ylabel(r'$T_{pd}$', fontsize=20)

# Set the scale font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)


plt.locator_params(axis='x', nbins=4)  # Control the number of x-axis ticks (adjustable as needed)
plt.locator_params(axis='y', nbins=4) # Control the number of scales on the y-axis (adjustable as needed)
plt.title(rf"$N = {N},\ g = {g},\ \delta = {delta}$", pad=20,fontsize=20)


plt.tight_layout()


plt.savefig('PLOT.pdf')


plt.show()


















