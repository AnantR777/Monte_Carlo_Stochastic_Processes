import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1A Monte Carlo Simulation - Paths x Timesteps

# Paths as ROWS!
# Timesteps as COLUMNS!
#         t0   t1    t2   ...
# path 1: (0, 0.1, 0.4, ...)
# path 2: (0, -0.3, 0.1, ...)

# Parameters
npaths = 20000 # Number of paths to be simulated
T = 1 # Time horizon
nsteps = 200 # Number of steps to over in [0,T]
dt = T/nsteps # Size of the timesteps
t = np.arange(0, T + dt, dt) # Define our time grid
mu = 0.12 #Mean/drift for our ABM
sigma = 0.4 # Vol/diffusion for our ABM

# Create an [npaths,nsteps] matrix to simulate the value at each time step along each path
#The formula for ABM is dX(t) = mu*dt + sigma*dW(t)
dX = mu*dt + sigma*np.sqrt(dt)*np.random.randn(npaths,nsteps)

# Cumulatively sum the values over the time steps to get each path
# Add a column of zeros at the beginning to represent the initial condition
X = np.hstack([np.zeros((npaths, 1)), np.cumsum(dX, axis=1)])

#Expected, mean and sample paths - Paths x Timesteps

EX = mu * t
#This will give us one path with an average value at each timestep, i.e. a [1,nsteps] vector.

# Plotting
plt.figure(1)
plt.plot(t, X[::1000, :].T, alpha=0.5)  # Sample paths (every 1000th path)
plt.plot(t, EX, 'r--', label='Expected path')  # Expected path
plt.plot(t, np.mean(X, axis=0), 'k--', label='Mean path')  # Mean path


# Setting labels, legend, and title
plt.legend()
plt.xlabel('time')
plt.ylabel('X')
plt.ylim([-1, 1])
plt.title(r'Arithmetic Brownian Motion $dX(t) = \mu dt + \sigma dW(t)$')

#Variance = Mean Square Deviation = Mean Square Displacement of Random Part
# From formula for ABM we know the random part: sigma*dW(t)
# So the square of this is: sigma^2*dt (since dW^2 = dt)

# Theoretical variance
theoretical_variance = sigma**2 * t

# Sampled variance
sampled_variance = np.var(X, axis=0)  # Variance of X along each time step

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

#Mean Absolute Deviation
# This is given by E(|X - EX|)
# Apparently if you compute this for ABM you reach a theoretical value of
# sigma*sqrt(2t/pi). Which is equivalent to sqrt(2*VarX / pi)
# Unfortunately I cannot get there, so we will have to take his word

# Theoretical mean absolute deviation
theoretical_mad = sigma * np.sqrt(2 * t / np.pi)

# Sampled mean absolute deviation
sampled_mad = np.mean(np.abs(X - EX), axis=0)  # Mean of the absolute deviation of X from EX

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

# Define the time horizons for histograms
horizon = np.array([50, 125, 200])  # Changed to match your nsteps
bmmin = np.min(X)
bmmax = np.max(X)
nbins = 100

##  Probability Distribution at different times
# Here we are plotting histograms at different times, to show how the
# probability distribution evolves (by considering the paths)
# Note the difference between the column number (20/80/end) and the ylabel,
# this is because we discretized our grid into 200 steps so at timestep 80
# we are 40% (0.4) of the way through to T=1

plt.figure(4, facecolor='white')

for j, ndays in enumerate(horizon):
    plt.subplot(3, 1, j+1)

    # Plotting the histogram for the ABM distribution at each time horizon
    plt.hist(X[:, ndays], bins=nbins, density=True, label='Simulated')

    # Generating data for the Gaussian curve based on the theoretical values
    sorted_data = np.linspace(bmmin, bmmax, 1000)  # Generating points for plotting
    gaussian_pdf = norm.pdf(sorted_data, EX[ndays], np.sqrt(theoretical_variance[ndays]))

    # Plotting the Gaussian curve
    plt.plot(sorted_data, gaussian_pdf, label='Gaussian')

    plt.xlabel('ABM Value')
    plt.legend(loc='upper left')
    plt.xlim([bmmin, bmmax])
    plt.title(f'Density at time {t[ndays]:.2f} (Years: {t[ndays]/T:.2f})')

plt.tight_layout()  # Adjusts the plots to ensure they don't overlap
plt.show()