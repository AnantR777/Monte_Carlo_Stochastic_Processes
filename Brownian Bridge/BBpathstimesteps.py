import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats
from scipy.stats import norm
from matplotlib.lines import Line2D

# Paths as ROWS!
# Timesteps as COLUMNS!
#         t0   t1    t2   ...
# path 1: (0, 0.1, 0.4, ...)
# path 2: (0, -0.3, 0.1, ...)

# Simulation of a Brownian Bridge using Monte Carlo Simulation
# ------------------------------------------------------------
# A Brownian Bridge is a stochastic process that describes Brownian motion
# conditioned to return to a specific endpoint at a specific time.

# Parameters:
# T = Total time duration
# nsteps = Number of steps in the simulation
# npaths = Number of independent paths to simulate
# start_point = Starting point of the Brownian Bridge
# end_point = Endpoint of the Brownian Bridge at time T

# The formula for Brownian Bridge is
#  dX(t) = (b-X)/(T-t) * dt + sigma*dW(t)

# The basic idea is to simulate standard Brownian motion and then adjust it
# to ensure it meets the endpoint condition.

# The BB presents a different type of SDE than what we have previously seen
# as it requires both a known start and end point. It then proceeds to
# simulate the different paths between those two points but its start and
# end will always be the same regardless of the path taken.
# This applies in Fixed Income, e.g. Bond pricing

# Parameters
npaths = 20000  # Number of paths to be simulated
T = 1  # Time Horizon
nsteps = 200  # Number of timesteps
dt = T / nsteps  # Size of our timesteps
t = np.linspace(0, T, nsteps + 1) # Discretization of our grid
a = 0.8  # Starting point
b = 1  # Ending point
sigma = 0.3  # Volatility/diffusion term

## Method 1: Monte Carlo Simulation via ABM

## Monte Carlo Simulation - npaths x nsteps - BB via ABM

# Due to some nice properties, we can simulate the BB via a driftless ABM
# i.e. A purely random ABM with the formula
# dX(t) = sigma*dW(t)

# We will use dW rather than dX to save X for our BB
# Simulate our ABM
dW = sigma * np.sqrt(dt) * np.random.randn(npaths, nsteps)

# Initialize W with the starting value 'a' and then add cumulative sum of dW
W = np.hstack([a * np.ones((npaths, 1)), dW]).cumsum(axis=1)
# W = np.zeros((npaths, nsteps + 1))
# W[:, 0] = a  # Set starting point to 'a'
# W[:, 1:] = np.cumsum(dW, axis=1) + a

# Brownian Bridge formula: X(t) = W(t) + (b - W(T)) / T * t
X = W + (b - W[:, -1][:, np.newaxis]) / T * t
X[:, -1] = b  # Ensuring the endpoint is exactly b


# In the above:
# repmat( b-W(:,end),1,nsteps+1) means take b subtract the last column of W
# for all rows and repeat it 1 times down (i.e. keep the same number of
# paths) and nsteps+1 times across (i.e. the number of timesteps) to form a
# [npaths,nsteps+1] matrix.
# AND
# repmat(t,npaths,1) means create a [1,t] vector with the time t
# as each column and do that npaths many rows to create a [npaths,nsteps+1]
# matrix

## Method 2: Monte Carlo Simulation via For Loop (Euler-Maruyama method)

## Monte Carlo Simulation -  npaths x nsteps - Traditional for loop

# Allocate and initialise all paths
X = np.zeros((npaths, nsteps + 1))
X[:, 0] = a  # Start point
X[:, -1] = b  # End point

# Compute the Brownian bridge with Euler-Maruyama
for i in range(nsteps):
    X[:,i + 1] = X[:,i] + (b - X[:,i]) / (nsteps - i + 1) + sigma * np.random.randn(npaths) * np.sqrt(dt)

print(X.shape)

## Expected, mean and sample paths

# The expected path below comes from Ballotta & Fusai p.135, where they
# have defined the E(X) on an interval [s,T], which is more general. In our
# case we have defined our interval to be [0,T], hence s=0 drops out and we
# are left with the formula below.
EX = a + (t * (b - a) / T)

# Plotting
plt.figure(1, figsize=(10, 6))
plt.plot(t,X[::1000,:].T); # Sample paths (every 1000th path)
plt.plot(t,EX, linewidth = 3, c = 'r', label='Expected path')  # Expected path
plt.plot(t, np.mean(X, axis = 0), label='Mean path')  # Mean path
plt.title('Brownian Bridge SDE Plot')
plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()

# The below are some additonal conditions that scale the picture according
# to some conditions - more of a nice to have.
sdevmax = sigma * np.sqrt(T) / 2
# plt.ylim([(a + b) / 2 - 4 * sdevmax, (a + b) / 2 + 4 * sdevmax])
plt.title('Brownian Bridge dX = ((b-X)/(T-t))dt + σdW')

# Theoretical variance
VARX = (sigma**2) * (t / T) * (T - t)

# Sampled variance (across paths at each timestep)
# Note the change in axis from 1 to 0
sampled_variance = np.var(X, axis=0)

# Mean square deviation
# Calculating the mean square deviation for each time step
# Again, note the change in axis from 1 to 0
mean_square_deviation = np.mean((X - EX[np.newaxis, :])**2, axis=0)

# Note that in Ballotta & Fusai they have no sigma term in the SDE for the
# BB (i.e. sigma=1). However, when they quote the variance then there is a
# sigma^2 missing from their figure (since sigma^2 = 1^2 = 1). But we have
# taken sigma to be a different value and therefore must multiply our
# theoretical variance formula by sigma^2.

# Plotting
plt.figure(2, figsize=(10, 6))
plt.plot(t, VARX, 'r', label='Theory')  # Theoretical variance
plt.plot(t, sampled_variance, 'm', label='Sampled')  # Sampled variance
plt.plot(t, mean_square_deviation, 'c--', label='Mean Square Deviation')  # Mean square deviation
plt.legend(loc='upper right')
plt.xlabel('Time (t)')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.title('Brownian Bridge Process: Variance')

# Mean Absolute Deviation
# Calculating the mean absolute deviation for each time step
mean_absolute_deviation = np.mean(np.abs(X - EX[np.newaxis, :]), axis=0)

# Plotting
plt.figure(3, figsize=(10, 6))
plt.plot(t, mean_absolute_deviation, 'g', label='Mean Absolute Deviation')
plt.legend(loc='upper right')
plt.xlabel('Time (t)')
plt.ylabel('MAD(X) = E(|X-E(X)|)')
plt.title('Brownian Bridge Process: Mean Absolute Deviation')


# NO theoretical closed form MAD for BB

# Computing the autocovariance for each path
unbiased_scale = nsteps - np.abs(np.arange(-nsteps + 1, nsteps))


# Initialize the autocovariance array
C = np.zeros((npaths, 2 * nsteps + 1))

# Calculate the autocovariance for each path
for i in range(npaths):
    deviation = X[i, :] - EX
    # Compute autocorrelation and normalize by the number of steps
    C[i, :] = correlate(deviation, deviation, mode='full') / nsteps

# Average the autocovariance across all paths
C_mean = np.mean(C, axis=0)

# Plotting the autocovariance
plt.figure(4, figsize=(10, 6))
plt.plot(C_mean[nsteps - 1:], 'r')  # Plot the second half of the autocovariance
plt.title('Autocovariance of Brownian Bridge')
plt.xlabel('Lag')
plt.ylabel('Autocovariance')

# Assuming the following variables are defined: nsteps, npaths, T

# Time points and range for x
t_points = np.array([0.05, 0.1, 0.2, 0.4, 1])
x = np.linspace(-3, 3, 200)

# Initialize arrays for analytical and sampled PDFs
fa = np.zeros((len(x), len(t_points)))  # Analytical
fs = np.zeros((len(x), len(t_points)))  # Sampled

# Define colors
colors = ['red', 'green', 'blue', 'orange', 'black']

# Generate sample paths
W_T = np.random.normal(0, np.sqrt(T), npaths)
t_grid = np.linspace(0, T, nsteps)
B_samples = np.zeros((npaths, nsteps))

for i in range(nsteps):
    t = t_grid[i]
    B_samples[:, i] = np.random.normal(0, np.sqrt(t), npaths) - t / T * W_T

# Compute the PDFs and plot
plt.figure(5)
for i, t in enumerate(t_points):
    color = colors[i]
    # Analytical PDF
    variance = t * (T - t) / T
    std_dev = np.sqrt(variance)
    fa[:, i] = norm.pdf(x, 0, std_dev)

    # Sampled PDF
    time_index = min(int(t / T * nsteps), nsteps - 1)
    bin_counts, _ = np.histogram(B_samples[:, time_index], bins=x, density=True)
    fs[:-1, i] = bin_counts

    plt.plot(x, fa[:, i], label=f't = {t:.2f} (Analytical)', color=color)
    plt.plot(x, fs[:, i], label=f't = {t:.2f} (Sampled)', linestyle='--', color=color)


# Create custom legend handles for time points
color_legend_handles = [Line2D([0], [0], color=colors[i], label=f'Time = {t_points[i]:.2f}') for i in range(len(t_points))]

# Add handles for "Analytical" and "Sampled" line types
color_legend_handles.append(Line2D([0], [0], color='k', linestyle='-', label='Analytical'))
color_legend_handles.append(Line2D([0], [0], color='k', linestyle='--', label='Sampled'))

# Create and set the combined legend
plt.legend(handles=color_legend_handles, loc='upper right')

plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.title('Probability Density Function of a Brownian Bridge at Different Times')


# Define the parameters for the Brownian Bridge
T = 1.0  # Total time
n = 1000  # Number of points

# Generate time points
t = np.linspace(0, T, n)

# Function to generate Brownian Bridge
def brownian_bridge(T, n):
    # Generate standard Brownian motion
    W = np.random.normal(0, np.sqrt(T/n), n).cumsum()
    # Adjust to Brownian Bridge
    B = W - (t/T) * W[-1]
    return B

# Generate multiple Brownian Bridge paths to create a distribution
num_paths = 10000
paths = np.array([brownian_bridge(T, n) for _ in range(num_paths)])

# Choose a specific time to view the distribution, e.g., t=0.5
specific_time = 0.5
index = int(n * specific_time / T)
values_at_time = paths[:, index]

# Plot the PDF at the specific time
plt.figure(figsize=(10, 6))
plt.hist(values_at_time, bins=50, density=True, alpha=0.6, color='g')

# Fit a normal distribution to the data
mu, std = stats.norm.fit(values_at_time)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = "PDF of Brownian Bridge at t={:.2f}".format(specific_time)
plt.title(title)
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()



