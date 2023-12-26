import numpy as np
import matplotlib.pyplot as plt

# Paths as ROWS!
# Timesteps as COLUMNS!
#         t0   t1    t2   ...
# path 1: (0, 0.1, 0.4, ...)
# path 2: (0, -0.3, 0.1, ...)

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

#  Generate a [npaths,nsteps] matrix of standard uniform random devaites
U = np.random.rand(npaths, nsteps)

# Convert those values in Bilateral Exponential (BE) random deviates
BE = -1/eta1 * np.log((1-U)/p) * (U >= 1-p) + 1/eta2 * np.log(U/(1-p)) * (U < 1-p)


## Monte Carlo Simulation - npaths x nsteps

# We calculate our traditional ABM of the form of the equation
dW = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(npaths, nsteps)

# Recall a Poisson Distribution ~Poi(lambda) can be interpreted by thinking
# of lambda as the expected number of events occuring. For instance,
# arrivals at a hospital in a certain hour can be modelled as a Poi(3)
# meaning we expect 3 people to arrive in any given hour. But of course it
# could be 1 (unlikley), 2 (more likely), right the way up to 10 and beyond
# (v. unlikely). They are all discrete though. So in our situation here,
# with lambda = 0.5, we are saying that we expect to jump about half the
# time, which means our values will be 0 (we don't jump) or 1 (we do jump)
# or potentially 2 on rare occasions (a v. big jump)

# We now need to compute an [npaths,nsteps] matrix of the jump points. That is the frequency of the jumps.
dN = np.random.poisson(lambda_ * dt, (npaths, nsteps))


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
X = np.hstack([np.zeros((npaths, 1)), np.cumsum(dX, axis=1)])

# Note this computes the paths of the log prices since we have used ABM
# To transform back to stock prices we require one final step
#S = S0*exp(X) ;

## Expected, mean and sample paths
# Calculate the expected path for the KJD process
# EX = (mu + lambda * (p/eta1 - (1-p)/eta2)) * t
EX = (mu + lambda_ * (p / eta1 - (1 - p) / eta2)) * t
print(EX.shape)
# Plotting
plt.figure(1)
plt.plot(t, EX, 'r', label='Expected path')  # Expected path
plt.plot(t, np.mean(X, axis=0), 'k', label='Mean path')  # Mean path
plt.plot(t, X[::1000].T, alpha=0.5)  # Sample paths (every 1000th path), semi-transparent

# Setting plot properties
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('X')
plt.ylim([-1, 1.2])
plt.title('Paths of a Kou Jump-Diffusion Process $X = \mu t + \sigma W(t) + \Sigma_{i=1}^{N(t)} Z_i$')

# Calculate the theoretical variance for the KJD process
# VARX = t * (sigma^2 + 2*lambda * (p/(eta1^2) + (1-p)/(eta2^2)))
VARX = t * (sigma**2 + 2 * lambda_ * (p / (eta1**2) + (1 - p) / (eta2**2)))

# Sample variance
sampled_variance = np.var(X, axis=0)

# Mean square deviation
mean_square_deviation = np.mean((X - EX[np.newaxis, :])**2, axis=0)

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

for j in range(npaths):
    # Calculate autocovariance for each path
    # Subtract EX from each path of X (broadcasting EX across all paths)
    path_diff = X[j, :] - EX
    autocov = np.correlate(path_diff, path_diff, mode='full') / (nsteps + 1)
    # Assign this autocovariance to the jth column of C
    C[:, j] = autocov

# Compute the mean across all paths
C_mean = np.mean(C, axis=1)


theoryAutoCov=(sigma**2+2*lambda_*(p/eta1**2+(1-p)/eta2**2))*t
fig, ax = plt.subplots()
ax.plot(t, theoryAutoCov, 'r', label = 'Theory for infinite t')
ax.plot(t, C_mean[nsteps:], 'g', label = 'Sampled')  # Plot the second half of C_mean
#With t=0
ax.plot(0, (sigma**2+2*lambda_*(p/eta1**2+(1-p)/eta2**2))*0, 'ro', label = 'Var for infinite t')
ax.plot(0, np.mean(np.var(X, axis=0)), 'go', label = 'Sampled Var')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$C(\tau)$')

ax.legend()
ax.set_title('Kou: autocovariance')

## Probability Density Function at different times

# Parameters for x-axis
dx = 0.02
x = np.arange(-1, 1, dx)
xx = x[:-1] + dx / 2  # Adjust to match the number of histogram values

# Select time points
time_points = [40, 100, -1]  # Corresponding to times 0.2, 0.5, and 1 in your grid
T = 1  # Assuming T is defined as the total time horizon

# Plotting
plt.figure(3, figsize=(10, 8))  # Set a larger figure size for clarity
for i, time_point in enumerate(time_points):
    plt.subplot(3, 1, i + 1)
    hist_values, _ = np.histogram(X[:, time_point], bins=x, density=True)
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

## Asymmetric double-sided exponential jump sizes - Simulate the Jump

#simulating the jump sizes according to the asymmetric double-sided exponential distribution,
#a key feature of Kou's approach to modeling financial asset dynamics.

# Asymmetric double-sided exponential distribution
# As used in S. G. Kou, A jump diffusion model for option pricing,
# Management Science 48, 1086-1101, 2002, https://doi.org/10.1287/mnsc.48.8.1086.166
# See also Ballotta and Fusai (2018), Section 6.2.2

# Parameters

# xmax = 2  # Truncation
# deltax = 0.01  # Grid step
# binw = 0.1  # Bin width
# n = 10**6  # Number of random samples

# # Compute the PDF
# x = np.arange(-xmax, xmax + deltax, deltax)  # Grid
# fX = p * eta1 * np.exp(-eta1 * x) * (x >= 0) + (1 - p) * eta2 * np.exp(eta2 * x) * (x < 0)  # PDF

# # Sample the distribution using inverse transform sampling
# U = np.random.rand(n)  # Standard uniform random variable
# X = -1 / eta1 * np.log((1 - U) / p) * (U >= 1 - p) + 1 / eta2 * np.log(U / (1 - p)) * (U < 1 - p)

# # Plot
# plt.figure(1)
# x2 = np.arange(-xmax, xmax + binw, binw)  # Bin edges for histogram
# plt.hist(X, bins=x2, density=True, alpha=0.7, label='Sampled')
# plt.plot(x, fX, linewidth=2, label='Theory')
# plt.xlabel('x')
# plt.ylabel('f_X')
# plt.legend()
# plt.title('Asymmetric Double-sided Distribution')

# # Display the plot
# plt.show()

## FFT
# Define the number of grid points for the FFT
N = 512  # A power of 2 is preferred for efficient FFT computation

# Define the grid step size in real space
dx = 0.1  # Spacing between individual points in real space

# Create a grid in real space
# np.arange creates an array of evenly spaced values within the specified range
# The grid is centered around zero (-N/2 to N/2) to facilitate FFT operations
x = dx * np.arange(-N/2, N/2)

# Define the grid step size in Fourier space (pulsation)
# The Nyquist relation dxi = 2*pi/(N*dx) is used to determine the spacing in Fourier space
dxi = (2 * np.pi) / (N * dx)
xi = dxi * np.arange(-N/2, N/2)  # Fourier space grid in pulsation space

# Define the grid step size in Fourier space (frequency)
dnu = 1 / (N * dx)  # Spacing in Fourier space for frequency domain
nu = dnu * np.arange(-N/2, N/2)  # Fourier space grid in frequency space

# Compute the exponent term of the characteristic function for the Kou Jump Diffusion
# This involves the bilateral exponential jump size distribution
expon = lambda_ * ((p * eta1) / (eta1 - 1j * xi) + ((1 - p) * eta2) / (eta2 + 1j * xi) - 1)

# Characteristic function of the Kou Jump Diffusion process
# This includes both the diffusion and jump components
char_func = np.exp((1j * xi * mu - 0.5 * (xi * sigma)**2) * T + expon * T)

# Compute the Fourier Transform using FFT
# np.fft.fft performs the fast Fourier transform
# np.fft.fftshift and np.fft.ifftshift are used to shift the zero-frequency component to/from the center
f_X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func))) / (N * dx)

# Repeat the process for frequency space
expon1 = lambda_ * ((p * eta1) / (eta1 - 1j * (2 * np.pi * nu)) + ((1 - p) * eta2) / (eta2 + 1j * (2 * np.pi * nu)) - 1)
char_func1 = np.exp((1j * (2 * np.pi * nu) * mu - 0.5 * ((2 * np.pi * nu) * sigma)**2) * T + expon1 * T)
f_X1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func1))) / (N * dx)

# f_X and f_X1 are the Fourier Transforms of the characteristic function of the Kou Jump Diffusion process
# in pulsation space and frequency space, respectively.

## Figures
plt.figure(4)
plt.plot(x, f_X.real, 'ko', linewidth=2, label='Re(fn)')
plt.plot(x, f_X.imag, 'ro', label='Im(fn)')
plt.hist(X[:, -1], bins=100, density=True)
plt.axis([-1.2, 1.2, 0, 1])
plt.title(r'Pulsation Space: FFT of KJD in $\xi$')
plt.xlabel('x')
plt.ylabel('f')
plt.legend()

plt.figure(5)
plt.plot(x, f_X1.real, 'ko', linewidth=2, label='Re(fn)')
plt.plot(x, f_X1.imag, 'ro', label='Im(fn)')
plt.hist(X[:, -1], bins=100, density=True)
plt.axis([-1.2, 1.2, 0, 1])
plt.title(r'Frequency Space: FFT of KJD in $\nu$')
plt.legend()

plt.show()
