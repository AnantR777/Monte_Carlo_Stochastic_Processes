import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

##Simulation of the Merton Jump Diffusion Process
# We now introduce stochastic processes with jumps, this means at a random
# point our process can jump up. This is generally considered a better way
# of modelling prices since empiracally they do tend to jump.

#  Our formula for the MJD is given in terms of X(t) rather than dX(t)
#  X(t) = (mu_S - 0.5*sigma_S^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i

# Note the above is our ABM for X(t), where X(t) is log(S/S0) i.e. the log
# of the stock price.

# The extra sum term in our solution is the random jump part.
# N(t) is a Poisson process, with arrival rate lamdba
# Z_i is our sequence of i.i.d random variables
# In the MJD process the random variables we will be using as our Z_i are
# normal distributions
# Unfortunately, since we are now dealing with two different distributions
# we need to distinguish between the parameters associated with each of
# them.

# We shall define:
# mu_S : the mean/drift of our traditional ABM (for simplicity muS)
# sigma_S : the vol/diffusion of our traditional ABM (again sigmaS)
# ... AND ...
# lambda : the rate of arrival for our Poisson Process
# mu_J : the mean/drift of our i.i.d Gaussian random variables (muJ)
# sigma_J : the vol/diffusion of our i.i.d Gaussians (sigmaJ)

#Parameters
npaths = 20000  # Number of paths to be simulated
T = 1  # Time horizon
nsteps = 200  # Number of timesteps
dt = T / nsteps  # Size of timesteps
t = np.linspace(0, T, nsteps + 1)  # Discretization of the grid
muS = 0.2  # Drift for ABM
sigmaS = 0.3  # Diffusion for ABM
lambda_ = 0.5  # Rate of arrival for Poisson Process
muJ = -0.1  # Drift for Jumps
sigmaJ = 0.15  # Diffusion for Jumps
S0 = 1  # Initial stock price

## Monte Carlo Simulation -  nsteps x npaths

# We calculate our traditional ABM of the form of the equation
# Step 1: Calculate the continuous part of the MJD process - Ballotta & Fusai p.178
dW = muS * dt + sigmaS * np.sqrt(dt) * np.random.randn(nsteps, npaths)

# Recall a Poisson Distribution ~Poi(lambda) can be interpreted by thinking
# of lambda as the expected number of events occuring. For instance,
# arrivals at a hospital in a certain hour can be modelled as a Poi(3)
# meaning we expect 3 people to arrive in any given hour. But of course it
# could be 1 (unlikley), 2 (more likely), right the way up to 10 and beyond
# (v. unlikely). They are all discrete though. So in our situation here,
# with lambda = 0.5, we are saying that we expect to jump about half the
# time, which means our values will be 0 (we don't jump) or 1 (we do jump)
# or potentially 2 on rare occasions (a v. big jump)

# We now need to compute an [nsteps,npaths] matrix of the jump points. That is the frequency of the jumps.

# Step 2: Simulate the jump frequency
dN = np.random.poisson(lambda_ * dt, (nsteps, npaths))

# Now we need to compute the size of the jumps.
# Step 3: Compute the jump sizes
dJ = muJ * dN + sigmaJ * np.sqrt(dN) * np.random.randn(nsteps, npaths)

# Here we are using the 'scale and shift' of our standard normal ~N(0,1) to
# get the scaled normal ~N(muS,sigmaS^2) which determines our jump sizes.

# Adding the two components together gives us the complete value at each timestep for the MJD process
# Step 4: Combine the continuous and jump components - Ballotta & Fusai p.178
dX = dW + dJ

# Cumulatively sum the rows to produce paths over time steps
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

# Note this computes the paths of the log prices since we have used ABM
# To transform back to stock prices we require one final step
S = S0 * np.exp(X)

## Expected, mean and sample paths

# Calculate the expected path for the MJD process
# EX = (muS + lambda * muJ) * t
EX = (muS + lambda_ * muJ) * t

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
plt.title('Paths of a Merton Jump-Diffusion Process $X = \mu t + \sigma W(t) + \Sigma_{i=1}^{N(t)} Z_i$')

## Variance

# Calculate the theoretical variance for MJD process
VARX = t * (sigmaS**2 + lambda_ * (muJ**2 + sigmaJ**2))

# Sample variance (now along paths, axis 1)
sampled_variance = np.var(X, axis=1)

# Mean square deviation (now along paths, axis 1)
mean_square_deviation = np.mean((X - EX[:, np.newaxis])**2, axis=1)

# Plotting
plt.figure(2)
plt.plot(t, VARX, 'r', label='Theory')  # Theoretical variance
plt.plot(t, sampled_variance, 'm', label='Sampled 1')  # Sampled variance
plt.plot(t, mean_square_deviation, 'c--', label='Sampled 2')  # Mean square deviation
plt.legend(loc='upper left')
plt.xlabel('Time (t)')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.ylim([0, 0.12])
plt.title('Merton Jump-Diffusion Process: Variance')

## Probability Density Function at different time

# Assuming X is already defined and has the shape (nsteps + 1, npaths)

# Parameters for x-axis
dx = 0.02
x = np.arange(-1, 1, dx)
xx = x[:-1] + dx / 2  # Shift required for bar chart

# Select time points
# Assuming the total time T is defined and t is the time array
time_points = [int(0.2 * len(t)), int(0.5 * len(t)), -1]  # Updated to use relative positions in the t array
labels = ['f_X(x,0.2)', 'f_X(x,0.5)', 'f_X(x,1)']

# Plotting
plt.figure(3)
for i, time_point in enumerate(time_points):
    plt.subplot(3, 1, i + 1)
    hist_values, _ = np.histogram(X[time_point, :], bins=x, density=True)  # Use X[time_point, :]
    plt.bar(xx, hist_values, width=dx)
    plt.xlim([-1, 1])
    plt.ylim([0, 3])
    plt.ylabel(labels[i])
    if i == len(time_points) - 1:
        plt.xlabel('x')

plt.suptitle('Probability Density Function of a Merton Jump-Diffusion Process at Different Times')

# Parameters
lamb = 1.5
mu = 0.03
sigma = 0.2
sigmaZ = 0.1
muZ = -0.03
expiry = 0.25

# Define range for possible jumps
N = np.arange(0, 31, 1)

# Define the grid for plotting
meanJD = (mu + lamb * muZ) * expiry
sdJD = np.sqrt((sigma**2 + lamb * sigmaZ**2) * expiry)
xmin = meanJD - 6 * sdJD
xmax = meanJD + 6 * sdJD
xT = np.linspace(xmin, xmax, 100)

# 1. Compute weights using Poisson distribution
weights = poisson.pmf(N, lamb * expiry)

# 2. Truncate weights if the cumulative sum reaches a threshold
cumsum_weights = np.cumsum(weights)
Nmax = np.max(np.where(cumsum_weights < 0.99999999)[0])

# 3. Calculate mean and variance for different numbers of jumps
meanJ = mu * expiry + np.arange(0, Nmax + 1, 1) * muZ
varJ = sigma**2 * expiry + np.arange(0, Nmax + 1, 1) * sigmaZ**2

# 4. Build the Merton density
pdfMerton = np.zeros(len(xT))
for j in range(Nmax + 1):
    pdfMerton += weights[j] * norm.pdf(xT, meanJ[j], np.sqrt(varJ[j]))

# Plotting
plt.figure(4)
plt.plot(xT, pdfMerton, label='Merton Pdf')
nor = norm.pdf(xT, meanJD, sdJD)
plt.plot(xT, nor, label='Gaussian Pdf')
plt.legend()
plt.title('Probability Density Function of the Merton Jump Diffusion Process')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()

