from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error ,num_epochs ,batch_size ,Truncate,set_num_sect
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
from tqdm import tqdm
import matplotlib



X = np.load("X.npy")

data = X[0, :, :]

# Get the time point, that is, multiply the column index by 0.01
time_points = np.arange(data.shape[1]) * 0.01



fig, ax = plt.subplots()

# Draw each row of data
for i in range(8):
    ax.plot(time_points, data[i, :],linewidth=2)


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
# Set axis labels and titles
ax.set_xlabel('Time', fontsize=23)
ax.set_ylabel('Activity', fontsize=23)


ax.tick_params(axis='both', which='major', labelsize=20)





plt.tight_layout()


fig.savefig('plot.pdf')


plt.show()