from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error, num_epochs, batch_size, Truncate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

J = np.load('J_cupy.npy')

r = 1

p = []
dw = []

# Early Stopping Parameters
patience = 100          # Number of epochs to wait for improvement
min_delta = 1e-6       # Minimum change to qualify as improvement

for i in tqdm(range(2, 100,2)):
    p.append(125 * i)
    q = np.zeros(r)

    for k in range(r):

        X = np.load("X.npy")
        X = torch.from_numpy(X[:i, :, :Truncate])
        X = torch.chunk(X, chunk, dim=2)
        X = torch.cat(X, dim=0)

        class DynamicModel(nn.Module):
            def __init__(self):
                super(DynamicModel, self).__init__()
                self.weight = nn.Parameter(torch.rand(N, N, dtype=torch.float64))

            def forward(self, trajectory):
                trajectory_point = trajectory
                reconstruct_trajectory = trajectory.clone()

                for p in range(reconstruct_trajectory.shape[2] - 1):
                    current_points = trajectory_point[:, :, p]
                    next_points = (1 - delta_t) * current_points + delta_t * torch.matmul(self.weight, torch.tanh(
                        trajectory_point[:, :, p].t())).t()

                    reconstruct_trajectory[:, :, p + 1] = next_points

                return reconstruct_trajectory

        model = DynamicModel()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        losses = []
        weight_losses = []

        train_dataset = TensorDataset(X, X)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        # Initialize Early Stopping Variables
        best_weight_loss = float('inf')
        epochs_no_improve = 0

        for j_epoch in range(10000):
            losss = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                loss = mean_squared_error(outputs, labels)
                loss.backward()  # Compute gradients
                optimizer.step()  # Update parameters
                losss += loss.item()

            losses.append(losss)
            current_weight_loss = np.mean((model.weight.data.clone().numpy() - J) ** 2)
            weight_losses.append(current_weight_loss)

            # Check for improvement
            if current_weight_loss + min_delta < best_weight_loss:
                best_weight_loss = current_weight_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # If no improvement for 'patience' epochs, stop training
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {j_epoch+1} for iteration {i}, run {k}")
                break

        q[k] = weight_losses[-1]
    dw.append(q)

print(dw)
#Add mean and standard deviation bars
mean_d_n = np.mean(dw, axis=1)
std_d_n = np.std(dw, axis=1)




plt.rcParams.update({'font.size': 25})
plt.plot(p, mean_d_n, linestyle='dashed', color='blue', linewidth=3)
plt.fill_between(p, mean_d_n - std_d_n, mean_d_n + std_d_n, color='blue', alpha=0.2)



plt.xlabel('$P$', fontsize=20)
plt.ylabel('$dW$', fontsize=20)

# 设置刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)




plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=4)
plt.title(rf"$N = {N},\ g = {g}$",  fontsize=20)

plt.tight_layout()


plt.savefig('ET5.pdf')


plt.show()









