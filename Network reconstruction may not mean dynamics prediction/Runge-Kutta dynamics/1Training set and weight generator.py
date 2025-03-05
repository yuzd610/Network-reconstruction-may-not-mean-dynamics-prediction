import numpy as np
from config import iterations, delta_t, N, g, set_num
from tqdm import tqdm

# Initialize the weight matrix J
J = np.random.normal(loc=0, scale=np.sqrt((g ** 2) / N), size=(N, N))
np.fill_diagonal(J, 0)

np.save('J_cupy.npy', J)

# Initialize the state matrix of all sample sets
X = np.random.normal(0, 1, size=(set_num, N, iterations + 1))


# Define the update function
def dx_dt(X_t):
    return -X_t + np.matmul(J, np.tanh(X_t).T).T


# Iterate and update each time step
for i in tqdm(range(iterations)):
    X_t = X[:, :, i]

    k1 = delta_t * dx_dt(X_t)
    k2 = delta_t * dx_dt(X_t + 0.5 * k1*delta_t)
    k3 = delta_t * dx_dt(X_t + 0.5 * k2*delta_t)
    k4 = delta_t * dx_dt(X_t + k3*delta_t)

    # Update formula: X_{t+1} = X_t + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    X[:, :, i + 1] = X_t + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Save the generated trajectory
np.save('X.npy', X)

