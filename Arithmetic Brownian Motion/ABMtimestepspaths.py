import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## 1B Monte Carlo Simulation - Timesteps x Paths

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

# Parameters
npaths = 20000 # Number of paths to be simulated
T = 1 # Time horizon
nsteps = 200 # Number of steps to over in [0,T]
dt = T/nsteps # Size of the timesteps
t = np.arange(0, T + dt, dt) # Define our time grid
mu = 0.12 #Mean/drift for our ABM
sigma = 0.4 # Vol/diffusion for our ABM

# Create an [nsteps, npaths] matrix to simulate the value at each time step along each path
#The formula for ABM is dX(t) = mu*dt + sigma*dW(t)
dX = mu * dt + sigma * np.sqrt(dt) * np.random.randn(nsteps, npaths)

# Cumulatively sum the values over the time steps to get each path
X = np.vstack([np.zeros(npaths), np.cumsum(dX, axis=0)])

# Note: `axis=0` in `np.cumsum` indicates we are adding each row to the previous one
# The 'vstack' with zeros is to account for the rows going downwards, starting from zero

# Expected, mean and sample paths - Timesteps x Paths

# Expected path
EX = mu * t  # The expected path, i.e., with no randomness dW

# Plotting
plt.figure(1, figsize=(10, 6))

# Plot a selection of paths
plt.plot(t, X[:, ::1000] , alpha = 0.5) # Sample paths (every 1000th path)

# Plot the expected path
plt.plot(t, EX, 'r--', label='Expected path')
# Plot the mean path
# Note: `axis=1` in `np.mean` to indicate we are taking an average of all paths (columns)
# at each timestep (rows). This will give us one path with an average value at each timestep.
plt.plot(t, np.mean(X, axis=1), 'k--', label='Mean path')

plt.legend()
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([-1, 1])
plt.title(r'Arithmetic Brownian motion dX(t) = µdt + $\sigma $dW(t)')

## 3B Variance = Mean Square Deviation = Mean Square Displacement of Random Part
# From formula for ABM we know the random part: sigma*dW(t)
# So the square of this is: sigma^2*dt (since dW^2 = dt)


# Theoretical variance
theoretical_variance = sigma**2 * t

# Sampled variance
sampled_variance = np.var(X, axis=1)  # Variance of X along each time step
# Note: `axis=1` in `np.var` to indicate we are computing variance of all paths (columns)
# at each timestep (rows).

# Plotting
plt.figure(2)
plt.plot(t, theoretical_variance, 'r', label=r'Theory: $\sigma²t$ = 2Dt')
# Here our 2Dt refers to the Fokker-Planck equation. D in that equation
# multiples the diffusion part and is set to 0.5*sigma^2 hence, subbing D
# into the above equation yields sigma^2*t giving equality.
plt.plot(t, sampled_variance, 'k', label='Sampled')

# Setting labels, legend, and title
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel('Var(X) = E((X-E(X))²)')
plt.title('Arithmetic Brownian Motion: Mean Square Displacement (MSD)')

## 4B Mean Absolute Deviation
# This is given by E(|X - EX|)
# Apparently if you compute this for ABM you reach a theoretical value of
# sigma*sqrt(2t/pi). Which is equivalent to sqrt(2*VarX / pi)
# Unfortunately I cannot get there, so we will have to take his word

# Theoretical mean absolute deviation
theoretical_mad = sigma * np.sqrt(2 * t / np.pi)

# Sampled mean absolute deviation
sampled_mad = np.mean(np.abs(X - EX[:, np.newaxis]), axis=1)  # Mean of the absolute deviation of X from EX
# Note: `axis=1` in `np.mean` to indicate we are computing mean of all paths (columns)
# at each timestep (rows).


# Plotting
plt.figure(3)
plt.plot(t, theoretical_mad, label='Theory: σ√(2t/π)')
plt.plot(t, sampled_mad, label='Sampled')

# Setting labels, legend, and title
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel('E(|X-E(X)|) = (2Var(X)/π)^(1/2)')
plt.ylim([0, 0.02])
plt.xlim([0,0.0045])
plt.title('Arithmetic Brownian Motion: Mean Absolute Deviation (MAD)')

bmmin = np.min(X)
bmmax = np.max(X)
nbins = 100
horizon = np.array([50, 125, 200])  # Changed to match your nsteps

# ## 5B Probability Distribution at different times
# # Here we are plotting histograms at different times, to show how the
# # probability distribution evolves (by considering the paths)
# # Note the difference between the column number (20/80/end) and the ylabel,
# # this is because we discretized our grid into 200 steps so at timestep 80
# # we are 40% (0.4) of the way through to T=1

plt.figure(4, facecolor='white')

for j, ndays in enumerate(horizon):
    plt.subplot(3, 1, j + 1)

    # Plotting the histogram
    plt.hist(X[ndays, :], bins=nbins, density=True, label='Simulated')

    # Sorting data for plotting the Gaussian curve
    sorted_data = np.sort(X[ndays, :])

    # Plotting the Gaussian curve
    plt.plot(sorted_data, norm.pdf(sorted_data, EX[ndays],
                                   np.sqrt(theoretical_variance[ndays])),
             label='Gaussian')

    plt.xlabel('ABM')
    plt.legend(loc='upper left')
    plt.xlim([bmmin, bmmax])
    plt.title(f'Density in {ndays / horizon[-1]:.2f} years')

plt.tight_layout()  # Adjusts the plots to ensure they don't overlap
plt.show()
