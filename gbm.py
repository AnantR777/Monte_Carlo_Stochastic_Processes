import numpy as np
import matplotlib.pyplot as plt

# The formula for ABM is
# dX(t) = mu*dt + sigma*dW(t)

#   The formula for GBM is
#   dS(t) = mu*S*dt + sigma*S*dW(t)
# However, using the transform X = log(S/S0) we can show that GBM is in
# fact ABM, with a = (mu - 0.5*simga^2) multiplying dt, and sigma
# multiplying dW.
# That is: dX = (mu - 0.5*sigma^2)*dt + sigma*dW
# We will use this result to compute our ABM (as before) and then transform
# back at the end using S = S0*exp(X)

# Parameters
npaths = 20000  # Number of paths to be simulated
T = 1  # Time horizon
nsteps = 200  # Number of steps to over in [0, T]
dt = T / nsteps  # Size of the timesteps
t = np.arange(0, T+dt, dt)  # Define our time grid
mu = 0.2  # Mean/drift for our ABM
sigma = 0.4  # Vol/diffusion for our ABM
S0 = 1  # Our initial stock price

# 1A Monte Carlo Simulation - Paths x Timesteps
# Paths as rows, timesteps as columns
# Accumulate the increments
# Now we need to cumulatively sum the values over the time steps to get
dX = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(npaths, nsteps)
# Accumulate the increments
# Now we need to cumulatively sum the values over the time steps to get
# each path
X = np.hstack((np.zeros((npaths, 1)), np.cumsum(dX, axis=1)))
S = S0 * np.exp(X)

# 1B Monte Carlo Simulation - Timesteps x Paths
# Timesteps as rows, paths as columns
dX = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(nsteps, npaths)
X = np.vstack((np.zeros((1, npaths)), np.cumsum(dX, axis=0)))
S_transposed = S0 * np.exp(X)

# 2A Expected, mean and sample paths - Paths x Timesteps
plt.figure(1)
ES = S0 * np.exp(mu * t)  # The expected path, i.e. with no randomness dW
plt.plot(t, ES, 'r.', label='Expected path')  # Expected path in red dots
plt.plot(t, np.mean(S, axis=0), 'k.', label='Mean path')  # Mean path in black dots

# Plotting a subset of paths (every 1000th path)
plt.plot(t, S[::1000, :].T, alpha=0.3)
# plots every 1000th row i.e. row 1000, 2000 etc.
# since there are 20000 rows we have 20 lines on the plot
# transpose because Matplotlib's plot function expects the columns as
# individual series to plot against the x-axis but we have rows atm

# Adjusting plot features
plt.legend()
plt.xlabel('t')
plt.ylabel('S')
plt.ylim([0, 2.5])
plt.title('Geometric Brownian motion dS = μSdt + σSdW')

# 2B Expected, mean and sample paths - Timesteps x Paths
plt.figure(2)
plt.plot(t, ES, 'r.', label='Expected path')  # Expected path in red dots
plt.plot(t, np.mean(S_transposed, axis=1), 'k.', label='Mean path')  # Mean path in black dots

# Plotting a subset of paths (every 1000th path)
plt.plot(t, S_transposed[:, ::1000], alpha=0.3)

# Adjusting plot features
plt.legend()
plt.xlabel('t')
plt.ylabel('S')
plt.ylim([0, 2.5])
plt.title('Geometric Brownian motion dS = μSdt + σSdW')

# 3A Variance
# Theoretical value of 2nd moment
ES2 = (S0**2) * np.exp(2 * t * mu + t * sigma**2)

# Theoretical value of Var(S)
VARS = ES2 - ES**2

# Reshape ES to have the same shape as S for broadcasting (ES as a row vector)
ES_reshaped = ES.reshape(1, -1)

# Recompute the sample variance
sample_variance = np.var(S, axis=0, ddof=1)  # Sample Variance with Bessel's correction
sample_variance_2 = np.mean((S - ES_reshaped)**2, axis=0)  # Sample Variance without Bessel's correction

# Plotting the theoretical variance, and two sampled variances
plt.figure(3)
plt.plot(t, VARS, 'r', label='Theory')  # Theoretical Variance in red
plt.plot(t, sample_variance, 'm', label='Sampled 1')  # Sample Variance in magenta
plt.plot(t, sample_variance_2, 'c--', label='Sampled 2')  # Sample Variance in cyan dashed

# Adding plot features
plt.legend(loc='lower right')  # Corrected location keyword
plt.xlabel('t')
plt.ylabel('Var(X) = E((X-E(X))^2)')
# plt.ylim([0, 0.0006])  # Uncomment to set y-axis limits if needed
plt.title('Geometric Brownian Motion: variance')

plt.figure(4)

# plot 3 graphs in a column
# Subplot for PDF at t=0.1 # 20th step out of 200 is time 0.1
plt.subplot(3, 1, 1)
plt.hist(S[:, 20], bins=np.arange(0, 3.5, 0.035), density=True)
plt.ylabel('f_X(x,0.15)')
plt.xlim([0, 3.5])
plt.ylim([0, 3.5])
plt.title('Geometric Brownian motion: PDF at different times')

# Subplot for PDF at t=0.4 # 80th step out of 200 is time 0.4
plt.subplot(3, 1, 2)
plt.hist(S[:, 80], bins=np.arange(0, 3.5, 0.035), density=True)
plt.xlim([0, 3.5])
plt.ylim([0, 3.5])
plt.ylabel('f_X(x,0.4)')

# Subplot for PDF at t=1 # 200th step out of 200 is time 1
plt.subplot(3, 1, 3)
plt.hist(S[:, -1], bins=np.arange(0, 3.5, 0.035), density=True)
plt.xlim([0, 3.5])
plt.ylim([0, 3.5])
plt.xlabel('x')
plt.ylabel('f_X(x,1)')

# Save the figure as a PNG image
plt.savefig('gbpdensities.png')

# Show the plot
plt.show()

