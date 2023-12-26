import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.signal import correlate

#  The formula for OUP is
#  dX(t) = alpha*(mu - X)*dt + sigma*dW(t)
# This could also be described as the Vasicek model, however, it is worth noting
# they are the same thing. Vasicek just applied OUP to finance,
# specifically interest rates.

# Parameters
npaths = 20000 # Number of paths to be simulated
T = 1 #Time horizon
nsteps = 200 # Number of steps in [0,T]
dt = T/nsteps #Time grid
t = np.arange(0,T + dt, dt) # Discretization of our time grid
alpha = 5 # Speed at which it is mean reverting
mu = 0.07 # Long run mean
sigma = 0.07 # Vol/diffusion term
X0 = 0.03 #Initial value (e.g. current interest rate)

##1A Monte Carlo Simulation - Paths x Timesteps

#  Paths as ROWS!
# Timesteps as COLUMNS!
#         t0   t1    t2   ...
# path 1: (0, 0.1, 0.4, ...)
# path 2: (0, -0.3, 0.1, ...)

# Since we now have our variable X with the equation (as part of the dt
# term) we can no longer simply compute an entre matrix for dX. We must use
# an iterative approach.

# Set up an [npaths,nsteps] matrix, with the first column all equal to X0
# (i.e. all paths start at X0) and the rest zeros (these will be filled out
# later).
X = np.zeros((npaths, nsteps + 1))
X[:, 0] = X0
print(X.shape)

# Define an [npaths,nsteps] matrix of normally distributed random numbers
N = np.random.randn(npaths,nsteps)
print(N.shape)


# # ----------------------------------------------
# # 1. Euler-Maruyama Method
# # ----------------------------------------------

# for i in range(nsteps):
#     X[:, i+1] = X[:, i] + alpha * (mu - X[:, i]) * dt + sigma * np.sqrt(dt) * N[:, i]

# ----------------------------------------------
# 2. Euler-Maruyama Method with Analytic Moments
# ----------------------------------------------
# To use the analytic moments method we need analytic expressions for the
# expectation E(X) and varaince Var(X)
# For the OUP we have these expressions (see Ballotta & Fusai p.94)

# E(X) = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) )
# Var(X) = (sigma^2/2*alpha) * ( 1-exp(-2*alpha*t) )

# We then ignore the form of our model and compute:
#dX = E(X) + np.sqrt(Var(X))*np.random.randn()
# Substituting our dt for t, and X0 with the X from the previous timestep

for i in range(nsteps):
    E_X = X[:, i] * np.exp(-alpha * dt) + mu * (1 - np.exp(-alpha * dt))
    Var_X = (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))
    X[:, i+1] = E_X + np.sqrt(Var_X) * N[:, i]


##########################################################################################################################################
## 2A Expected, mean and sample paths and long term average

# Calculate the expected path
EX = X0 * np.exp(-alpha * t) + mu * (1 - np.exp(-alpha * t))

# Begin plotting

plt.figure(1)
plt.plot(t, X[::1000, :].T, alpha=0.5)  # Plotting sample paths
plt.plot(t, EX, 'r-', label='Expected path',linewidth=2)
plt.plot(t, np.mean(X, axis=0), 'k--', label='Mean path',linewidth=2, markersize=6)
plt.plot(t, mu * np.ones_like(t), 'k:', label='Long-term average')

plt.legend()
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([mu - 4 * (sigma / np.sqrt(2 * alpha)), mu + 4 * (sigma / np.sqrt(2 * alpha))])
plt.title(r'Ornstein-Uhlenbeck process $dX = \alpha(\mu-X)dt + \sigma dW$')

##########################################################################################################################################
## 3A Variance = Mean Square Deviation

# Calculate the theoretical variance of the Ornstein-Uhlenbeck process
# This follows the formula: sigma^2 / (2 * alpha) * (1 - exp(-2 * alpha * t))

VARX = sigma**2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))

# Plotting
plt.figure(2, figsize=(10, 6))

# Plot the theoretical variance as a red line
plt.plot(t, VARX, 'r', label='Theory')

# Plot the linear growth of variance over time (sigma^2 * t) as a green line
plt.plot(t, sigma**2 * t, 'g', label=r'$\sigma^2t$')

# Plot the asymptotic limit of variance (sigma^2 / (2 * alpha)) as a blue line
plt.plot(t, sigma**2 / (2 * alpha) * np.ones_like(t), 'b', label=r'$\sigma^2/(2\alpha)$')

# Plot the sampled variance (computed from the simulated paths) as a magenta line
# np.var calculates variance across rows (for each timestep), with ddof=1 for an unbiased estimator
plt.plot(t, np.var(X, axis=0, ddof=1), 'm', label='Sampled 1')

# Plot the mean squared deviation (Sampled 2) as a cyan dashed line
# This is calculated as the mean of the squared differences from the expected path
plt.plot(t, np.mean((X - EX)**2, axis=0), 'c--', label='Sampled 2')

plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel(r'Var(X) = E$((X-E(X))^2)$')
plt.ylim([0, 0.0006])
plt.title(r'Ornstein-Uhlenbeck process: variance')

##########################################################################################################################################
##4A Mean Absolute Deviation


# Plotting
plt.figure(3, figsize=(10, 6))

# Calculate and plot the theoretical Mean Absolute Deviation (MAD) as a red line
# The formula for theoretical MAD in the Ornstein-Uhlenbeck process is:
# sigma * sqrt((1 - exp(-2 * alpha * t)) / (pi * alpha))

plt.plot(t, sigma * np.sqrt((1 - np.exp(-2 * alpha * t)) / (np.pi * alpha)), 'r', label='Theory')

# Plot the linear growth of MAD over time (sigma * sqrt(2 * t / pi)) as a green line
plt.plot(t, sigma * np.sqrt(2 * t / np.pi), 'g', label=r'$\sigma(2t/\pi)^{1/2}$')

# Plot the asymptotic limit of MAD (sigma / sqrt(pi * alpha)) as a blue line
plt.plot(t, sigma / np.sqrt(np.pi * alpha) * np.ones_like(t), 'b', label='Long-term average')

# Plot the sampled Mean Absolute Deviation (MAD) as a magenta line
# This is calculated as the mean of the absolute deviations from the expected path
plt.plot(t, np.mean(np.abs(X - EX), axis=0), 'm', label='Sampled')

# Add legend, labels, and set y-axis limits
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel(r'E$|X-E(X)| = (2Var(X)/\pi)^{1/2}$')
plt.ylim([0, 0.02])

# Title of the plot, using LaTeX for mathematical symbols
plt.title('Ornstein-Uhlenbeck process: mean absolute deviation')


##########################################################################################################################################
#### 5A Autocovariance

# Initialize the autocovariance array
C = np.zeros((npaths, 2 * nsteps + 1))

# Calculate the autocovariance for each path
for i in range(npaths):
    deviation = X[i, :] - EX
    # Compute autocorrelation and normalize by the number of steps
    C[i, :] = correlate(deviation, deviation, mode='full') / nsteps



# Average over all paths
C = np.mean(C, axis=0)

# Begin plotting
plt.figure(4, figsize=(10, 6))

# Plot theoretical autocovariance
# The formula is: sigma^2 / (2 * alpha) * exp(-alpha * t)
plt.plot(t, sigma**2 / (2 * alpha) * np.exp(-alpha * t), 'r', label='Theory')

# Plot sampled autocovariance
# Only plotting the second half since it's symmetric and the first half corresponds to negative lags
plt.plot(t, C[nsteps:], 'b', label='Sampled')


# Plot variance for infinite t (sigma^2 / (2 * alpha))
plt.plot(0, sigma**2 / (2 * alpha), 'ro', label ='Theory Variance for Infinite t')  # Var for infinite t

# Plot average sampled variance
plt.plot(0, np.mean(np.var(X, axis=0, ddof=0)), 'bo', label='Average Sampled Var', linewidth=1.5)


# Add labels, legend, and title
plt.xlabel(r'$\tau$')
plt.ylabel(r'C($\tau$)')
plt.legend(loc='lower left')
plt.title('Ornstein-Uhlenbeck process: autocovariance')


##########################################################################################################################################
## 6A Autocorrelation

# The autocorrelation is the Covariance/Variance. However, since our OUP is
# only quasi-stationary (i.e. it is only stationary in the limit t -> inf)
# we will compute the autocorrelation as we have done above, in the limit
# as t -> inf

# It can be shown that in the limit, the autocorrelation becomes
# Corr(t,s) = exp(-1*alpha*tau)     with t < s

# Theoretical autocorrelation
CORRX = np.exp(-alpha * t)

# Begin plotting
plt.figure(5, figsize=(10, 6))

# Plot theoretical autocorrelation as a red line
plt.plot(t, CORRX, 'r', label='Theory')

# Plot sampled autocorrelation
# The sampled autocorrelation is computed as the autocovariance divided by the variance at time 0 (C[nsteps+1])
# Here, we use the second half of C (corresponding to non-negative lags)
plt.plot(t, C[nsteps:] / C[nsteps], 'b', label='Sampled')

# Add labels, legend, and title
plt.xlabel(r'$\tau$')
plt.ylabel(r'c($\tau$)')
plt.legend()
plt.title('Ornstein-Uhlenbeck process: autocorrelation')


##########################################################################################################################################
## 7A PDF At Different Times
# Calculate the standard deviation in the limit as t -> infinity

sdev_infinity = sigma / np.sqrt(2 * alpha)

# Define time points for which to calculate and plot the PDF
t2 = np.array([0.05, 0.1, 0.2, 0.4, 1])

# Calculate the expected value and standard deviation at these time points
EX2 = mu + (X0 - mu) * np.exp(-alpha * t2)
sdev = sigma * np.sqrt((1 - np.exp(-2 * alpha * t2)) / (2 * alpha))

# Set up space for the PDF
x = np.linspace(-0.02, mu + 4 * sdev_infinity, 200)


# Arrays to hold the analytical and sampled PDFs
fa = np.zeros((len(x), len(t2)))  # analytical
fs = np.zeros((len(x), len(t2)))  # sampled

# Define a set of colors for plotting
colors = plt.cm.viridis(np.linspace(0, 1, len(t2)))

# Calculate the analytical and sampled PDFs
for i, time in enumerate(t2):
    fa[:, i] = norm.pdf(x, EX2[i], sdev[i])
    bin_counts, _ = np.histogram(X[:, int(time * nsteps)], bins=x, density=True)
    fs[:-1, i] = bin_counts

# Begin plotting
plt.figure(6, figsize=(10, 6))

# Plot analytical and sampled PDFs
for i, color in enumerate(colors):
    plt.plot(x, fa[:, i], color=color, linestyle='-', label=f'{t2[i]:.2f}' if i == 0 else "")
    plt.plot(x, fs[:, i], color=color, linestyle='--')

# Create custom legend handles for time points
color_legend_handles = [Line2D([0], [0], color=colors[i], label=f'Time = {t2[i]:.2f}') for i in range(len(t2))]

# Add handles for "Analytical" and "Sampled" line types
color_legend_handles.append(Line2D([0], [0], color='k', linestyle='-', label='Analytical'))
color_legend_handles.append(Line2D([0], [0], color='k', linestyle='--', label='Sampled'))

# Create and set the combined legend
plt.legend(handles=color_legend_handles, loc='upper right')


# Add labels and title
plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.title('Ornstein-Uhlenbeck process: PDF at different times')

# Display the plot
plt.show()
