import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import lognorm
from matplotlib.lines import Line2D

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

## 1B Monte Carlo Simulation - Timesteps x Paths

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

# Create an [nsteps,npaths] matrix to simulate the value at each time step
# along each path
dX = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(nsteps, npaths)


# Now we need to cumulateively sum the values over the time steps to get each path
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

# Transform back to Geometric Brownian Motion (GBM)
S = S0 * np.exp(X)


############################################################################################################################################
## 2B Expected, mean and sample paths - Timesteps x Paths


# Plotting

ES = S0 * np.exp(mu * t)  # The expected path, i.e. with no randomness dW
plt.figure(1)
plt.plot(t, S[:, ::1000], alpha=0.5)  # Sample paths (first 1000 paths), semi-transparent
plt.plot(t, np.mean(S, axis=1), 'k.', label='Mean path')  # Mean path
plt.plot(t, ES, 'r.', label='Expected path')  # Expected path

# Setting plot properties
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('S')
plt.ylim([0, 2.5])
plt.title('Geometric Brownian Motion $dS = \mu S dt + \sigma S dW$')

############################################################################################################################################
## 3B Variance

# Calculate the theoretical second moment for GBM
ES2 = (S0**2) * np.exp(2 * mu * t + sigma**2 * t)

# Calculate the theoretical variance of S
VARS = ES2 - ES**2

# Sample variance - using axis=1 since paths are now columns
sampled_variance = np.var(S, axis=1)

# Mean square deviation - using axis=1 and adjusting the broadcasting of ES
mean_square_deviation = np.mean((S - ES[:, np.newaxis])**2, axis=1)

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
for i, t_value in enumerate(t2):
    color = colors[i]
    time_index = np.searchsorted(t,
                                 t_value)  # Find the index of the closest time step

    # Empirical PDF from the simulated data
    # Since S is transposed, we take the row corresponding to the time index
    hist, bin_edges = np.histogram(S[time_index, :], bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, hist, label=f't = {t_value:.2f} (Empirical)',
             linestyle='--', color=color)

    # Theoretical PDF using log-normal distribution
    scale = np.exp((mu - 0.5 * sigma ** 2) * t_value)
    shape = sigma * np.sqrt(t_value)
    s = np.linspace(0.01, S0 * 3, 200)
    theoretical_pdf = lognorm.pdf(s, s=shape, scale=scale)
    plt.plot(s, theoretical_pdf, label=f't = {t_value:.2f} (Theoretical)',
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
plt.xlim(0, S0 * 3)  # Adjust x-axis limits if necessary
plt.title('Probability Density Function of GBM at Different Times')
plt.show()