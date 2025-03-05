import numpy as np
from config import iterations, delta_t, N, g, set_num
from tqdm import tqdm

# Initialize the weight matrix J
J = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))

np.fill_diagonal(J, 0)

np.save('J_cupy.npy', J)

# Initialize the state matrix of all sample sets
X = np.random.normal(0, 1, size=(set_num, N, iterations + 1))


J_scaled = delta_t * J

# Iterate and update each time step
for i in tqdm(range(iterations)):
    X[:, :, i + 1] = (1 - delta_t) * X[:, :, i] + np.matmul(J_scaled, np.tanh(X[:, :, i]).T).T


np.save('X.npy', X)
