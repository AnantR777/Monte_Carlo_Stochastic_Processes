import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.signal import correlate
from matplotlib.lines import Line2D
from scipy.stats import lognorm

## Geometric Brownian Motion simulation
#   The formula for GBM is
#   dS(t) = mu*S*dt + sigma*S*dW(t)
# However, using the transform X = log(S/S0) we can show that GBM is in
# fact ABM, with a = (mu - 0.5*sigma^2) multiplying dt, and sigma
# multiplying dW.
# That is: dX = (mu - 0.5*sigma^2)*dt + sigma*dW
# We will use this result to compute our ABM (as before) and then transform
# back at the end using S = S0*exp(X)


# --------------------------------------------------
# Simulate our ABM
# --------------------------------------------------

# Parameters
npaths = 20000  # Number of paths to be simulated
T = 1  # Time horizon
nsteps = 200  # Number of steps in [0, T]
dt = T / nsteps  # Size of the timesteps
t = np.linspace(0, T, nsteps + 1)  # Time grid
mu = 0.2  # Mean/drift for our ABM
sigma = 0.4  # Volatility/diffusion for our ABM
S0 = 1  # Initial stock price

## 1A Monte Carlo Simulation - Paths x Timesteps

# Paths as ROWS!
# Timesteps as COLUMNS!
#         t0   t1    t2   ...
# path 1: (0, 0.1, 0.4, ...)
# path 2: (0, -0.3, 0.1, ...)

# Create an [npaths,nsteps] matrix to simulate the value at each time step
# along each path
# dX = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * N(0,1)
dX = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(npaths, nsteps)

# Cumulatively sum the values over time steps to get each path
# Adding a column of zeros at the beginning for the initial value
X = np.hstack([np.zeros((npaths, 1)), np.cumsum(dX, axis=1)])

# Transform back to Geometric Brownian Motion (GBM)
# S = S0 * exp(X)
S = S0 * np.exp(X)

############################################################################################################################################
## 2A Expected, mean and sample paths - Paths x Timesteps


# Calculate the expected path for GBM
ES = S0 * np.exp(mu * t)

# Plotting
plt.figure(1)
plt.plot(t, ES, 'r.', label='Expected path')  # Expected path
plt.plot(t, np.mean(S, axis=0), 'k.', label='Mean path')  # Mean path
plt.plot(t, S[::1000].T, alpha=0.5)  # Sample paths (every 1000th path), semi-transparent

# Setting plot properties
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('S')
plt.ylim([0, 2.5])
plt.title('Geometric Brownian Motion $dS = \mu S dt + \sigma S dW$')

############################################################################################################################################
## 3A Variance


# Calculate the theoretical second moment for GBM
ES2 = (S0**2) * np.exp(2 * mu * t + sigma**2 * t)

# Calculate the theoretical variance of S
VARS = ES2 - ES**2

# Sample variance
sampled_variance = np.var(S, axis=0)

# Mean square deviation
mean_square_deviation = np.mean((S - ES[np.newaxis, :])**2, axis=0)

# Plotting
plt.figure(2)
plt.plot(t, VARS, 'r', label='Theory')  # Theoretical variance
plt.plot(t, sampled_variance, 'm', label='Sampled 1')  # Sampled variance
plt.plot(t, mean_square_deviation, 'c--', label='Sampled 2')  # Mean square deviation
plt.legend(loc='lower right')
plt.xlabel('Time (t)')
plt.ylabel('Var(S) = E((S-E(S))^2)')
plt.title('Geometric Brownian Motion: Variance')

# Time points and range for the stock price (S)
t2 = np.array([0.05, 0.1, 0.2, 0.4, 1])

# Define colors for plotting
colors = ['red', 'green', 'blue', 'orange', 'black']

# Compute and plot the PDFs
plt.figure(3, figsize=(10, 6))
for i, t in enumerate(t2):
    color = colors[i]
    time_index = int(t / T * nsteps)

    # Empirical PDF from the simulated data
    hist, bin_edges = np.histogram(S[:, time_index], bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, hist, label=f't = {t:.2f} (Empirical)',
             linestyle='--', color=color)

    # Theoretical PDF using log-normal distribution
    scale = np.exp((mu - 0.5 * sigma ** 2) * t)
    shape = sigma * np.sqrt(t)
    s = np.linspace(0.01, S0 * 3, 200)
    theoretical_pdf = lognorm.pdf(s, s=shape, scale=scale)
    plt.plot(s, theoretical_pdf, label=f't = {t:.2f} (Theoretical)',
             color=color)

# Create custom legend handles for time points
legend_handles = [Line2D([0], [0], color=color, label=f'Time = {t2[i]:.2f}') for
                  i, color in enumerate(colors)]
legend_handles += [
    Line2D([0], [0], color='k', linestyle='-', label='Theoretical')]
legend_handles += [
    Line2D([0], [0], color='k', linestyle='--', label='Empirical')]

# Set the combined legend
plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel('Stock Price (S)')
plt.ylabel('Probability Density')
plt.xlim(0, 2.5)
plt.title('Probability Density Function of GBM at Different Times')
plt.show()

