
from config import iterations, delta_t, N, g, set_num, chunk, mean_squared_error ,num_epochs ,batch_size ,Truncate,set_num_sect
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20
J = np.load('J_cupy.npy')
J_study = np.load('J_study.npy')


X =[]





for i in range(200):
    for j in range(200):
        X.append(np.array([J[i][j], J_study[i][j]]))

X = np.array(X)



distances = (X[:,1]-X[:,0])**2
#Sorting
indices = np.argsort(distances)[-40000:]

# Extract these points
X= X[indices]





# Create the figure and axes
fig, ax = plt.subplots()

# Draw a scatter plot
sc = ax.scatter(X[:, 0], X[:, 1], linewidth=0.8)




# Set axis labels
ax.set_xlabel("$J_{ij}^1$", fontsize=22)
ax.set_ylabel("$J_{ij}^2$", fontsize=22)

# Set the font size of the scale
ax.tick_params(axis='both', labelsize=20)


plt.tight_layout()


plt.savefig('dia.pdf')


plt.show()