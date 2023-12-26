import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats
from scipy.stats import norm
from matplotlib.lines import Line2D

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

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

## Monte Carlo Simulation -  nsteps x npaths - BB via ABM

# Due to some nice properties, we can simulate the BB via a driftless ABM
# i.e. A purely random ABM with the formula
# dX(t) = sigma*dW(t)

# We will use dW rather than dX to save X for our BB
# Simulate our ABM


dW = sigma * np.sqrt(dt) * np.random.randn(nsteps, npaths)

# Initialize W with the starting value 'a' and then add cumulative sum of dW
# Transpose the structure to nsteps x npaths
W = np.vstack([a * np.ones((1, npaths)), dW]).cumsum(axis=0)

# Brownian Bridge formula: X(t) = W(t) + (b - W(T)) / T * t
# Adjust for the transposed structure
X = W + (b - W[-1, :]) / T * t[:, np.newaxis]
X[-1, :] = b  # Ensuring the endpoint is exactly b

# # In the above:
# # repmat( b-W(:,end),1,nsteps+1) means take b subtract the last column of W
# # for all rows and repeat it 1 times down (i.e. keep the same number of
# # paths) and nsteps+1 times across (i.e. the number of timesteps) to form a
# # [npaths,nsteps+1] matrix.
# # AND
# # repmat(t,npaths,1) means create a [1,t] vector with the time t
# # as each column and do that npaths many rows to create a [npaths,nsteps+1]
# # matrix

## Method 2: Monte Carlo Simulation via for loop (Euler Maruyama method)

## Monte Carlo Simulation - nsteps x npaths - Traditional for loop

# We need to initialise our matrix such that the start and end points are a
# and b respectively. Note we use (nsteps-1) zeros in between.

#Allocate and initialise all paths
X = np.vstack((a*np.ones(npaths), np.zeros((nsteps - 1,npaths)), b*np.ones(npaths)))

# Compute the Brownian bridge with Euler-Maruyama
for i in range(nsteps):
    X[i + 1, :] = X[i, :] + (b - X[i, :]) / (nsteps - i + 1) + sigma * np.random.randn(npaths) * np.sqrt(dt)

## Expected, mean and sample paths

# The expected path below comes from Ballotta & Fusai p.135, where they
# have defined the E(X) on an interval [s,T], which is more general. In our
# case we have defined our interval to be [0,T], hence s=0 drops out and we
# are left with the formula below.
EX = a + ( t * (b-a)/T)

# Plotting
plt.figure(1, figsize=(10, 6))
plt.plot(t,X[:,::1000]); # Sample paths (every 1000th path)
plt.plot(t,EX, linewidth = 3, c = 'r', label='Expected path')  # Expected path
plt.plot(t, np.mean(X, axis = 1), label='Mean path')  # Mean path
plt.title('Brownian Bridge SDE Plot')
plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()

# The below are some additonal conditions that scale the picture according
# to some conditions - more of a nice to have.
sdevmax = sigma * np.sqrt(T) / 2
plt.ylim([(a + b) / 2 - 4 * sdevmax, (a + b) / 2 + 4 * sdevmax])
plt.title('Brownian Bridge dX = ((b-X)/(T-t))dt + σdW')

## Variance = Mean Square Deviation

# Theoretical variance
VARX = (sigma**2) * (t / T) * (T - t)

# Sampled variance (across paths at each timestep)
sampled_variance = np.var(X, axis=1)

# Mean square deviation
# Calculating the mean square deviation for each time step
mean_square_deviation = np.mean((X - EX[:, np.newaxis])**2, axis=1)


# Note that in Ballotta & Fusai they have no sigma term in the SDE for the
# BB (i.e. sigma=1). However, when they quote the variance then there is a
# sigma^2 missing from their figure (since sigma^2 = 1^2 = 1). But we have
# taken sigma to be a different value and therefore must multiply our
# theoretical variance formula by sigma^2.


# Plotting
plt.figure(2)
plt.plot(t, VARX, 'r', label='Theory')  # Theoretical variance
plt.plot(t, sampled_variance, 'm', label='Sampled 1')  # Sampled variance
plt.plot(t, mean_square_deviation, 'c--', label='Sampled 2')  # Mean square deviation
plt.legend(loc='lower right')
plt.xlabel('Time (t)')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.title('Brownian Bridge Process: Variance')

# Autocovariance
EX = a + (b - a) * t / T
C = np.zeros((2 * nsteps + 1, npaths))
for j in range(npaths):
    C[:, j] = np.correlate(X[:, j] - EX, X[:, j] - EX, mode='full') / (
                nsteps + 1)  # unbiased estimator
# sampled Cov:
C = np.mean(C, axis=1)

# Plotting autocovariance
plt.figure(3)
plt.plot(t, C[nsteps:], 'r', label='Sampled')
plt.xlabel('t')
plt.ylabel('C(t)')
plt.title('Brownian Bridge Process: autocovariance')
plt.legend()
plt.show()
