import numpy as np
import matplotlib.pyplot as plt

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

# Simulation of the Kou Jump Diffusion Process using Monte Carlo Simulation
# -------------------------------------------------------------------------
# The Kou Jump Diffusion Process (KJD) is a stochastic model used in finance
# to capture the behavior of asset prices with jumps. It's an extension of
# standard diffusion processes by incorporating both continuous and discrete
# components, capturing the sudden jumps often observed in financial markets.
# This follows the same approach as the MJD process but uses a different random
# variable as the i.i.d components of the jumps.

# Where the MJD used Gaussians, we will now use the Bilateral Exponential
# distribution. This is a minor modification of the Laplace (or double
# exponential) Distribution as it is no longer symmetric down the y-axis.
# That is we have different exponential distributions for x>0 and for x<0,
# reflecting the fact that prices tend to be asymmetric.

# We follow the same approach as for the MJD, and display the KJD in its
# X(t) form:
# X(t) = (mu - 0.5 * sigma^2)* t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i

# Note the above is our ABM for X(t), where X(t) is log(S/S0) i.e. the log
# of the stock price.

# Parameters:
# T = Total time duration of the simulation
# nsteps = Number of time steps in the simulation
# npaths = Number of independent simulation paths
# start_point = Initial value of the asset
# mu = Drift coefficient of the process
# sigma = Volatility (standard deviation of the process)
# lambda = Jump intensity (frequency of jumps) / the rate of arrival for our Poisson Process
# eta1 = Parameter for upward jump size distribution (Bilateral Exponential)  -  This means the upward jumps have mean 1/eta1
# eta2 = Parameter for downward jump size distribution (Bilateral Exponential) - # This means the downward jumps have mean -1/eta2
# p = Probability of an upward jump (0 <= p <= 1)

# The Kou Jump Diffusion Process is described by the SDE:
# dX(t) = mu * dt + sigma * dW(t) + dJ(t)

# where:
# - X(t) is the log of the asset price at time t
# - dW(t) represents the Wiener process component (standard Brownian motion)
# - dJ(t) represents the jump component, modeled using a compound Poisson process
#   with jump sizes following a Bilateral Exponential distribution

# Key Features:
# - The process combines Gaussian diffusion with jumps, providing a more
#   realistic model of asset price dynamics compared to pure diffusion models.
# - The Bilateral Exponential distribution for jump sizes allows asymmetric behavior,
#   reflecting real market scenarios where upward and downward movements may have
#   different characteristics.

# In financial modeling, the KJD is used for:
# - Modeling stock prices and indices which exhibit sudden, large movements.
# - Pricing options and other derivatives where jump risks are significant.
# - Risk management, especially in portfolios sensitive to large market moves.

# Simulating a KJD involves generating random paths that follow the above
# SDE. This requires simulating both the continuous Brownian motion and
# the discrete jump process, usually using the Euler-Maruyama method for
# the continuous part and a Poisson process for the jumps.

# The simulation produces a set of paths for the asset price, capturing
# both normal market fluctuations and significant jumps, offering a more
# nuanced view of potential price trajectories

# Parameters
npaths = 20000  # Number of paths to be simulated
T = 1  # Time horizon
nsteps = 200  # Number of timesteps
dt = T / nsteps  # Size of timesteps
t = np.linspace(0, T, nsteps + 1)  # Discretization of the time grid
mu = 0.2  # Drift for ABM
sigma = 0.3  # Volatility/diffusion term for ABM . increasing can increase size of jumps
lambda_ = 0.5  # Rate of arrival for Poisson Process. increasing will increase the number of jumps
eta1 = 6  # Parameter for upward jumps
eta2 = 8  # Parameter for downward jumps
p = 0.4  # Probability of an upward jump
S0 = 1  # Initial stock price

## Generating the Bilateral Exponential Random Deviates
# Additional parameters for the bilateral exponential random deviates
muJ = -0.1
sigmaJ = 0.15

# Generate a [nsteps,npaths] matrix of standard uniform random deviates
U = np.random.rand(nsteps, npaths)

# Convert those values in Bilateral Exponential (BE) random deviates
BE = -1/eta1 * np.log((1-U)/p) * (U >= 1-p) + 1/eta2 * np.log(U/(1-p)) * (U < 1-p)

# Calculate the continuous part of the ABM equation
dW = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(nsteps, npaths)

# Recall a Poisson Distribution ~Poi(lambda) can be interpreted by thinking
# of lambda as the expected number of events occuring. For instance,
# arrivals at a hospital in a certain hour can be modelled as a Poi(3)
# meaning we expect 3 people to arrive in any given hour. But of course it
# could be 1 (unlikley), 2 (more likely), right the way up to 10 and beyond
# (v. unlikely). They are all discrete though. So in our situation here,
# with lambda = 0.5, we are saying that we expect to jump about half the
# time, which means our values will be 0 (we don't jump) or 1 (we do jump)
# or potentially 2 on rare occasions (a v. big jump)

# Compute an [nsteps,npaths] matrix of the jump points (frequency of the jumps)
dN = np.random.poisson(lambda_ * dt, (nsteps, npaths))


# Now we need to compute the size of the jumps.
# This is simply computing the size of the jumps (given by matrix BE) and
# when they occur (given by matrix dN)
# Its output will be a matrix that has components 0 (no jump) or some
# value (the size of the jump)
dJ = dN * BE

# Adding the two components together gives us the complete value at each
# timestep for the KJD process
dX = dW + dJ


# Cumulatively sum the increments to get the log price paths
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

# Note this computes the paths of the log prices since we have used ABM
# To transform back to stock prices we require one final step
#S = S0*exp(X) ;

## Expected, mean and sample paths
# Calculate the expected path for the KJD process
# EX = (mu + lambda * (p/eta1 - (1-p)/eta2)) * t
EX = (mu + lambda_ * (p / eta1 - (1 - p) / eta2)) * t

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
plt.ylim([-1, 1.2])
plt.title('Paths of a Kou Jump-Diffusion Process $X = \mu t + \sigma W(t) + \Sigma_{i=1}^{N(t)} Z_i$')

# Calculate the theoretical variance for the KJD process
# VARX = t * (sigma^2 + 2*lambda * (p/(eta1^2) + (1-p)/(eta2^2)))
VARX = t * (sigma**2 + 2 * lambda_ * (p / (eta1**2) + (1 - p) / (eta2**2)))

# Sample variance (now along paths, axis 1)
sampled_variance = np.var(X, axis=1)

# Mean square deviation (now along paths, axis 1)
mean_square_deviation = np.mean((X - EX[:, np.newaxis])**2, axis=1)

# Plotting
plt.figure(2)
plt.plot(t, VARX, 'r', label='Theory')  # Theoretical variance
plt.plot(t, sampled_variance, 'm', label='Sampled 1')  # Sampled variance
plt.plot(t, mean_square_deviation, 'c--', label='Sampled 2')  # Mean square deviation
plt.legend(loc='upper right')
plt.xlabel('Time (t)')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.title('Kou Jump-Diffusion Process: Variance')

# Autocovariance
C = np.zeros((2 * nsteps + 1, npaths))  # Preallocate autocovariance array

for i in range(npaths):
    # Calculate autocovariance for each path
    path_diff = X[:, i] - EX  # Corrected to use the transposed structure
    autocov = np.correlate(path_diff, path_diff, mode='full') / (nsteps + 1)
    C[:, i] = autocov

# Compute the mean across all paths
C_mean = np.mean(C, axis=1)

# Theoretical Autocovariance for infinite t
theoryAutoCov = (sigma**2 + 2 * lambda_ * (p / eta1**2 + (1 - p) / eta2**2)) * t

# Plotting
fig, ax = plt.subplots()
ax.plot(t, theoryAutoCov, 'r', label='Theory for infinite t')
ax.plot(t, C_mean[nsteps:], 'g', label='Sampled')  # Plot the second half of C_mean
# With t=0
ax.plot(0, (sigma**2 + 2 * lambda_ * (p / eta1**2 + (1 - p) / eta2**2)) * 0, 'ro', label='Var for infinite t')
ax.plot(0, np.mean(np.var(X, axis=1)), 'go', label='Sampled Var')  # Correct axis for variance
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$C(\tau)$')
ax.legend()
ax.set_title('Kou Jump-Diffusion: Autocovariance')

## Probability Density Function at different times

# Parameters for x-axis
dx = 0.02
x = np.arange(-1, 1, dx)
xx = x[:-1] + dx / 2  # Adjust to match the number of histogram values

# Select time points
time_points = [40, 100, -1]  # Corresponding to specific times in the simulation grid
T = 1  # Total time horizon

# Plotting
plt.figure(figsize=(10, 8))  # Set a larger figure size for clarity
for i, time_point in enumerate(time_points):
    plt.subplot(3, 1, i + 1)
    hist_values, _ = np.histogram(X[time_point, :], bins=x, density=True)  # Use X[time_point, :]
    plt.bar(xx, hist_values, width=dx)
    plt.xlim([-1, 1])
    plt.ylim([0, 3])
    plt.ylabel('PDF')
    if i == len(time_points) - 1:
        plt.xlabel('x')
    # Adding a title to each subplot
    current_time = t[time_point]
    plt.title(f'PDF of Kou Jump-Diffusion Process at t = {current_time:.2f} (Years: {current_time/T:.2f})')

plt.suptitle('Probability Density Function of a Kou Jump-Diffusion Process at Different Times')
plt.tight_layout()  # Adjust layout for better presentation

## Simulate the Jump

#simulating the jump sizes according to the asymmetric double-sided exponential distribution,
#a key feature of Kou's approach to modeling financial asset dynamics.

# Asymmetric double-sided exponential distribution
# As used in S. G. Kou, A jump diffusion model for option pricing,
# Management Science 48, 1086-1101, 2002, https://doi.org/10.1287/mnsc.48.8.1086.166
# See also Ballotta and Fusai (2018), Section 6.2.2

# Parameters
eta1 = 4
eta2 = 3
p = 0.4
xmax = 2  # Truncation
deltax = 0.01  # Grid step
binw = 0.1  # Bin width
n = 10**6  # Number of random samples

# Compute the PDF
x = np.arange(-xmax, xmax + deltax, deltax)  # Grid
fX = p * eta1 * np.exp(-eta1 * x) * (x >= 0) + (1 - p) * eta2 * np.exp(eta2 * x) * (x < 0)  # PDF

# Sample the distribution using inverse transform sampling
U = np.random.rand(n)  # Standard uniform random variable
X = -1 / eta1 * np.log((1 - U) / p) * (U >= 1 - p) + 1 / eta2 * np.log(U / (1 - p)) * (U < 1 - p)

# Plot
plt.figure(figsize=(10, 6))
x2 = np.arange(-xmax, xmax + binw, binw)  # Bin edges for histogram
plt.hist(X, bins=x2, density=True, alpha=0.7, label='Sampled')
plt.plot(x, fX, linewidth=2, color='red', label='Theory')
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend()
plt.title('Asymmetric Double-sided Exponential Distribution in Kou\'s Model')
plt.grid(True)  # Added grid for better readability
plt.xlim([-xmax, xmax])  # Set x-axis limits to match the theoretical range

# Display the plot
plt.show()
