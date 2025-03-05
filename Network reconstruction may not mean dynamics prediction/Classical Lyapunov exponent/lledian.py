
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tqdm import tqdm


ε = 0.000001
delta_t = 0.01
step_t = 10000
N = 1000
T = step_t * delta_t
n = 5
def random_unit_vector(N):
    # Generate an N-dimensional random vector
    random_vector = np.random.randn(N)

    # Calculate the modulus of the vector
    norm = np.linalg.norm(random_vector)

    # Normalize vector
    unit_vector = random_vector / norm

    return unit_vector



#Generate two trajectories

X_n_1 = np.zeros((N, step_t + 1))
X_n_2 = np.zeros((N, step_t + 1))


unit_vector = random_unit_vector(N)

X_n_1[:, 0] =  np.random.normal(0, 1, N)
X_n_2[:, 0] =  X_n_1[:, 0] + ε*unit_vector







LLE = []
G = []

for g in tqdm(np.arange(0, 5, 0.2)):
    λ = np.zeros(n)
    for i in tqdm(range(n)):
        D = []
        J_1 = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))
        np.fill_diagonal(J_1, 0)
        for t in range(0, step_t):
            X_n_1[:, t + 1] = (1 - delta_t) * X_n_1[:, t] + delta_t * np.matmul(J_1, np.tanh(X_n_1[:, t]))
            X_n_2[:, t + 1] = (1 - delta_t) * X_n_2[:, t] + delta_t * np.matmul(J_1, np.tanh(X_n_2[:, t]))

            a = np.log(np.sqrt(np.sum((X_n_2[:, t + 1] - X_n_1[:, t + 1]) ** 2)) / ε) / delta_t
            D.append(a)
            X_n_2[:, t + 1] = X_n_1[:, t + 1] + ε * (X_n_2[:, t + 1] - X_n_1[:, t + 1]) / (
                np.linalg.norm((X_n_2[:, t + 1] - X_n_1[:, t + 1])))
        λ[i]= np.mean(D[2000:])
    LLE.append(λ)
    G.append(g)













#Generate mean and standard deviation bars

mean_LLE = np.mean(LLE, axis=1)
std_LLE = np.std(LLE, axis=1)

# Plotting









plt.plot(G, mean_LLE, linestyle='dashed', color='red',linewidth=3)  # 设置为蓝色虚线
plt.fill_between(G, mean_LLE - std_LLE, mean_LLE + std_LLE, color='red', alpha=0.2)
# Set the axis labels and specify the font size
plt.xlabel('g ', fontsize=30)
plt.ylabel('λ', fontsize=30)

# Set the scale font size
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)

# Add a legend and specify the font size
plt.axvline(x=1, linestyle='dashed', color='black', linewidth=2, label='g = 1')

# Adjust the size and style of the tick marks
plt.tick_params(axis='both', which='major', labelsize=30, length=10, width=2)

# Control the tick interval to ensure that the tick labels do not overlap
plt.locator_params(axis='x', nbins=5)  # 控制x轴的刻度数量（可根据需要调整）
plt.locator_params(axis='y', nbins=4)  # 控制y轴的刻度数量（可根据需要调整）


plt.tight_layout()
plt.savefig('PLOT3.pdf')

plt.show()