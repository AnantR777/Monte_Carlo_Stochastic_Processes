import numpy as np
import matplotlib.pyplot as plt

# Another of our jump diffusion processes is the KJD. This follows the same
# approach as the MJD process but uses a different random variable as the
# i.i.d components of the jumps.
# Where the MJD used Gaussians, we will now use the Bilateral Exponential
# distribution. This is a minor modification of the Laplace (or double
# exponential) Distribution as it is no longer symmetric down the y-axis.
# That is we have different exponential distributions for x>0 and for x<0,
# reflecting the fact that prices tend to be asymmetric.

# We follow the same approach as for the MJD, and display the KJD in its
# X(t) form:
# X(t) = (mu - 0.5*sigma^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i

# Note the above is our ABM for X(t), where X(t) is log(S/S0) i.e. the log
# of the stock price.

# Let us again define our parameters:
# mu : the mean/drift of our traditional ABM
# sigma : the vol/diffusion of our traditional ABM
# ... AND ...
# lambda : the rate of arrival for our Poisson Process
# eta1 : the upward jump parameter of Bilat. Exp. random variables
# This means the upward jumps have mean 1/eta1
# eta2 : the downward jump parameter of our i.i.d Bilat. Exp.
# This means the downward jumps have mean 1/eta2
# p : the probability of a jump for our i.i.d Bilat. Exp.

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
# S = S0*exp(X) ;

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
# C = np.zeros((2*nsteps+1, npaths))

# # Autocovariance (slightly modified for matching)
# C = np.zeros((2*nsteps+1, npaths))
# for j in range(npaths):
#     C[:,j] = np.correlate(X[:,j]-EX, X[:,j]-EX, mode='full')/(nsteps+1) # unbiased estimator
# C = np.mean(C, axis=1)

# C = np.mean(C, axis=1)
# fig, ax = plt.subplots()
# ax.plot(t, VARX, 'r')
# ax.plot(t, C[nsteps:], 'g')
# ax.plot(0, VARX[0], 'go')
# ax.plot(0, np.mean(np.var(X, axis=0)), 'bo')
# ax.set_xlabel(r'$\tau$')
# ax.set_ylabel(r'$C(\tau)$')
# ax.legend(['Theory for infinite t', 'Sampled', 'Var for infinite t', 'Sampled Var'])
# ax.set_title('Kou: Autocovariance')

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
plt.figure(figsize=(10, 8))  # Set a larger figure size for clarity
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
plt.figure(5)
x2 = np.arange(-xmax, xmax + binw, binw)  # Bin edges for histogram
plt.hist(X, bins=x2, density=True, alpha=0.7, label='Sampled')
plt.plot(x, fX, linewidth=2, label='Theory')
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend()
plt.title('Asymmetric Double-sided Distribution')

# Display the plot
plt.show()