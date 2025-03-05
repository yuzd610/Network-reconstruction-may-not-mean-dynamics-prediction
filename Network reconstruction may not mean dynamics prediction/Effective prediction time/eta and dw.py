import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 200
g = 0.8


d_W = np.linspace(0, 2 * g**2 / N, 500)

# Define functions
def eta(N, g, d_W):
    term = 1 - N * d_W / (2 * g**2)
    return np.sqrt(1 - term**2)

# Calculate the function value
eta_values = eta(N, g, d_W)

# Draw the graphics
plt.rc('font', family='serif', size=12)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=30)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.rc('legend', fontsize=12)


plt.figure(figsize=(8, 6))
plt.plot(d_W, eta_values, color="blue", linewidth=3)

# Set axis labels
plt.xlabel(r"$d_W$")
plt.ylabel(r"$\eta$")
plt.title(rf"$N = {N},\ g = {g}$", pad=20)

# Add reference lines
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

# Add legend and grid

plt.grid(visible=True, linestyle="--", linewidth=0.5)

# Optimize scale
plt.tick_params(axis='both', which='major', length=6, width=1, direction='in')  # 主刻度
plt.tick_params(axis='both', which='minor', length=4, width=0.5, direction='in')  # 次刻度


plt.tight_layout()

plt.savefig('2.pdf')
plt.show()