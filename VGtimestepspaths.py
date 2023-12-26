import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

## Simulation of the Variance Gamma Process

# The VG Process is again different to the SDEs we've seen before. Instead
# of traditional SDEs (such as GBM or OUP) and their extension to include
# random jumps (as seen in MJD or KJD) we now consider time changed
# Brownian Motion. That is we take our traditional ABM but instead of it
# evolving over normal ("calendar") time, we have it evolve over random
# time using a so called "random clock".

#  The formula for our new stochastic process is
#  X(t) = theta*G(t) + sigma*W(G(t))

# For our purposes we will define G(t) to be a Gamma Process with the following parameters:
# alpha = lambda = 1/k, where k >0
# Where we have chosen the above such that:
# E(X) = t : this means our random clock evolves in line with calendar time on average
# Var(X) = kt

# We will now work with the VG in its differential form
#  dX(t) = theta*dG(t) + sigma*dW(G(t))

# Parameters
npaths = 20000  # Number of paths to be simulated
T = 1  # Time horizon
nsteps = 200  # Number of timesteps
dt = T / nsteps  # Size of the timesteps
t = np.linspace(0, T, nsteps + 1)  # Discretization of the time grid
theta = 0.2  # Drift term for our time-changed process
sigma = 0.3  # Volatility/diffusion term for our time-changed process
kappa = 0.05  # Parameter for the Gamma Process = 1/lambda = 1/rate

## Monte Carlo Simulation

# Step 1: Compute the Gamma increments for the random clock
dG = gamma.rvs(dt/kappa, scale=kappa, size=(nsteps, npaths))

# Step 2: Compute the VG process increments under the Gamma random clock
dX = theta * dG + sigma * np.sqrt(dG) * np.random.randn(nsteps, npaths)

# Step 3: Cumulatively sum the increments to get the VG process paths
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

## Expected, mean and sample paths

# Calculate the expected path for the VG process
EX = theta * t

# Plotting
plt.figure(1)
plt.plot(t, EX, 'r', label='Expected path')  # Expected path
plt.plot(t, np.mean(X, axis=1), 'k', label='Mean path')  # Mean path

# Sample paths (every 1000th path), semi-transparent
for i in range(0, npaths, 1000):
    plt.plot(t, X[:, i], alpha=0.5)

# Setting plot properties
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('X')
plt.ylim([-0.8, 1.2])
plt.title('Paths of a Variance Gamma Process $dX(t) = \\theta dG(t) + \\sigma dW(G(t))$')

# Calculate the theoretical variance for the VG process
VARX = t * (sigma**2 + theta**2 * kappa)

# Sample variance (calculated along paths, axis 1)
sampled_variance = np.var(X, axis=1)

# Mean square deviation (calculated along paths, axis 1)
mean_square_deviation = np.mean((X - EX[:, np.newaxis])**2, axis=1)

# Plotting
plt.figure(2)
plt.plot(t, VARX, 'r', label='Theory')  # Theoretical variance
plt.plot(t, sampled_variance, 'm', label='Sampled 1')  # Sampled variance
plt.plot(t, mean_square_deviation, 'c--', label='Sampled 2')  # Mean square deviation
plt.legend(loc='upper right')
plt.xlabel('Time (t)')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.title('Variance Gamma Process: Variance')

## pdf of a VG process at diff times

# Parameters for x-axis
dx = 0.02
x = np.arange(-0.8, 1.2, dx)
xx = x[:-1] + dx / 2  # Adjust to match the number of histogram values

# Select time points
time_points = [40, 100, -1]  # Corresponding to specific times in the simulation grid
labels = ['f_X(x,0.2)', 'f_X(x,0.5)', 'f_X(x,1)']

# Plotting histograms
plt.figure(3)
for i, time_point in enumerate(time_points):
    plt.subplot(3, 1, i + 1)
    hist_values, _ = np.histogram(X[time_point, :], bins=x, density=True)  # Use X[time_point, :]
    plt.plot(xx, hist_values, drawstyle='steps-mid')
    plt.xlim([-0.8, 1.2])
    plt.ylim([0, 4])
    plt.ylabel(labels[i])
    if i == len(time_points) - 1:
        plt.xlabel('x')
plt.suptitle('PDF of a Variance Gamma Process at Different Times - Histograms')

# Plotting bar charts
plt.figure(4)
for i, time_point in enumerate(time_points):
    plt.subplot(3, 1, i + 1)
    hist_values, _ = np.histogram(X[time_point, :], bins=x, density=True)
    plt.bar(xx, hist_values, width=dx)
    plt.xlim([-0.8, 1.2])
    plt.ylim([0, 4])
    plt.ylabel(labels[i])
    if i == len(time_points) - 1:
        plt.xlabel('x')
plt.suptitle('PDF of a Variance Gamma Process at Different Times - Bar Charts')

# Display the plot
plt.show()
