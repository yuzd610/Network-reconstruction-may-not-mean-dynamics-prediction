import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 23  # 可以统一设置字体大小
g= 2


delta_t = 0.01


N = 200
n = 5

Disturb = []
ε = []




X_n_1 = np.zeros((N, 2))
X_n_2 = np.zeros((N, 2))

for i in tqdm(np.arange(0, 1, 0.1)):

    Disturb.append(i)
    λ = np.zeros(n)
    for j in tqdm(range(n)):
        var_12 = np.sqrt(1 - i ** 2)


        X_n_1[:, 0] = np.random.normal(0, 1, N)
        X_n_2[:, 0] = X_n_1[:, 0]

        J_1 = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))
        J_11 = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))

        np.fill_diagonal(J_1, 0)
        np.fill_diagonal(J_11, 0)
        J_2 = var_12 * J_1 + J_11 * i
        X_n_1[:, 1] = (1 - delta_t) * X_n_1[:, 0] + delta_t * np.matmul(J_1, np.tanh(X_n_1[:, 0]))
        X_n_2[:, 1] = (1 - delta_t) * X_n_2[:, 0] + delta_t * np.matmul(J_2, np.tanh(X_n_2[:, 0]))
        c =np.sqrt( np.sum((X_n_1[:, 1] - X_n_2[:, 1]) ** 2))
        λ[j] =c
    ε.append(λ)






#Generate mean and standard deviation bars

mean_d_n = np.mean(ε, axis=1)
std_d_n = np.std(ε, axis=1)




plt.plot(Disturb, mean_d_n, linestyle='dashed', color='blue',linewidth=3)  # 设置为蓝色虚线
plt.fill_between(Disturb, mean_d_n - std_d_n, mean_d_n + std_d_n, color='blue', alpha=0.2)
# Set the axis labels and specify the font size

plt.xlabel(r"$\eta$", fontsize=23)
plt.ylabel(r"${\varepsilon ^{\frac{1}{2}}}$", fontsize=23)
plt.title(rf"$N = {N},\ g = {g}$", pad=20)
# Set the scale font size
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)



plt.tight_layout()
plt.savefig('eat =2.pdf')

plt.show()




