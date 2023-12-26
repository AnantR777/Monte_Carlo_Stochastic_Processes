import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

#  The formula for our Brownian Bridge is
#  dX(t) = (b-X)/(T-t) *dt + sigma*dW(t)
# The BB presents a different type of SDE than what we have previously seen
# as it requires both a known start and end point. It then proceeds to
# simulate the different paths between those two points but its start and
# end will always be the same regardless of the path taken.
# This applies in Fixed Income, e.g. Bond pricing

# Define parameters and time grid
npaths = 20000  # number of paths
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.linspace(0, T, nsteps + 1)  # observation times
sigma = 0.3  # volatility
a = 0.8  # initial value
b = 1  # final value

## Monte Carlo method 1

# We need to initialise our matrix such that the start and end points are a
# and b respectively.

# Allocate and initialise all paths
X = np.zeros((nsteps + 1, npaths))
# first path:
X[0, :] = a
# last path:
X[-1, :] = b

# Compute the Brownian bridge with Euler-Maruyama
for i in range(nsteps):
    # xrisimopia to Dx
    X[i + 1, :] = X[i, :] + (b - X[i, :]) / (
                nsteps - i + 1) + sigma * np.random.randn(npaths) * np.sqrt(dt)

## Monte Carlo method 2:

# Compute the increments of driftless arithmetic Brownian motion
# dW = sigma*np.random.randn((nsteps, npaths))*np.sqrt(dt)
# Accumulate the increments of arithmetic Brownian motion
# W = np.cumsum(np.vstack([a*np.ones((1,npaths)), dW]), axis=0)
# Compute the Brownian bridge with X(t) = W(t) + (b-W(T))/T*t
# X = W + np.tile((b-W[-1,:])/T*t, (npaths,1)).T


# Mean:_______
# Expected, mean and sample paths
# The expected path below comes from Ballotta & Fusai p.135, where they
# have defined the E(X) on an interval [s,T], which is more general. In our
# case we have defined our interval to be [0,T], hence s=0 drops out and we
# are left with the formula below.

EX = a + (b - a) * t / T  # expected path
EX = EX[:, None]  # correct shape for broadcasting

plt.figure(1)
plt.plot(t, EX, 'k')
plt.plot(t, np.mean(X, axis=1), ':k')
plt.plot(t, X[:, ::1000], alpha=0.4)
plt.legend(['Expected path', 'Mean path'])
plt.xlabel('t')
plt.ylabel('X')
plt.title('Brownian bridge dX = ((b-X)/(T-t))dt + sigmadW')

# Variance = Mean Square Deviation
plt.figure(2)
VARX = (sigma**2) * (t/T) * (T - t)
# Plotting theoretical variance, sampled variance using numpy's var function, and sampled variance using mean of squared deviations
plt.plot(t, VARX, 'r', label='Theory')
plt.plot(t, np.var(X, axis=1), 'm', label='Sampled 1')
plt.plot(t, np.mean((X - EX)**2, axis=1), 'c--', label='Sampled 2')
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.title('Brownian Bridge Process: variance')
plt.show()

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