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

## 1B Monte Carlo Simulation - Timesteps x Paths

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

# Initialize the array with timesteps as rows and paths as columns
X = np.zeros((nsteps + 1, npaths))
X[0, :] = X0

# Define an [nsteps, npaths] matrix of normally distributed random numbers
N = np.random.randn(nsteps, npaths)


# # ----------------------------------------------
# # 1. Euler-Maruyama Method
# # ----------------------------------------------
# for i in range(nsteps):
#     X[i+1, :] = X[i, :] + alpha * (mu - X[i, :]) * dt + sigma * np.sqrt(dt) * N[i, :]


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
    E_X = X[i, :] * np.exp(-alpha * dt) + mu * (1 - np.exp(-alpha * dt))
    Var_X = (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))
    X[i+1, :] = E_X + np.sqrt(Var_X) * N[i, :]



##########################################################################################################################################
## 2B Expected, mean and sample paths and long term average
# Calculate the expected path
# For the Ornstein-Uhlenbeck process, the expected path formula is:
# EX = X0 * exp(-alpha * t) + mu * (1 - exp(-alpha * t))
EX = X0 * np.exp(-alpha * t) + mu * (1 - np.exp(-alpha * t))

# Begin plotting
plt.figure(1, figsize=(10, 6))

# Plot the expected path as red dots
plt.plot(t, EX, 'r.', label='Expected path')

# Plot the mean path (average across all paths at each timestep) as black dots
plt.plot(t, np.mean(X, axis=1), 'k.', label='Mean path')

# Plot the long-term average (mu) as a black dashed line
plt.plot(t, mu * np.ones_like(t), 'k--', label='Long-term average')

# Plot sample paths (every 1000th path for clarity)
plt.plot(t, X[:, ::1000], alpha=0.5)  # Reduced alpha for less emphasis

# Add legend, labels, and set y-axis limits
plt.legend()
plt.xlabel('t')
plt.ylabel('X')

# Compute the standard deviation as t -> infinity
sdev_infinity = sigma / np.sqrt(2 * alpha)
plt.ylim([mu - 4 * sdev_infinity, mu + 4 * sdev_infinity])

# Title of the plot, using LaTeX for mathematical symbols
plt.title(r'Ornstein-Uhlenbeck process $dX = \alpha(\mu-X)dt + \sigma dW$')


##########################################################################################################################################
## 3B Variance = Mean Square Deviation

# Calculate theoretical variance of the Ornstein-Uhlenbeck process
# The formula for theoretical variance is: sigma^2 / (2 * alpha) * (1 - exp(-2 * alpha * t))
VARX = sigma**2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))

# Begin plotting
plt.figure(2, figsize=(10, 6))

# Plot the theoretical variance as a red line
plt.plot(t, VARX, 'r', label='Theory')

# Plot the linear growth of variance over time (sigma^2 * t) as a green line
plt.plot(t, sigma**2 * t, 'g', label=r'$\sigma^2t$')

# Plot the asymptotic limit of variance (sigma^2 / (2 * alpha)) as a blue line
plt.plot(t, sigma**2 / (2 * alpha) * np.ones_like(t), 'b', label=r'$\sigma^2/(2\alpha)$')

# Plot the sampled variance (Sampled 1) as a magenta line
# np.var computes variance along columns (for each timestep)
plt.plot(t, np.var(X, axis=1, ddof=1), 'm', label='Sampled 1')

# Plot mean square deviation (Sampled 2) as a cyan dashed line
# This is calculated as the mean of the squared differences from the expected path
plt.plot(t, np.mean((X - EX[:, None])**2, axis=1), 'c--', label='Sampled 2')

# Add legend, labels, and set y-axis limits
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel(r'Var(X) = E$((X-E(X))^2)$')
plt.ylim([0, 0.0006])

# Title of the plot, using LaTeX for mathematical symbols
plt.title('Ornstein-Uhlenbeck process: variance')


##########################################################################################################################################
##4B Mean Absolute Deviation

# Begin plotting
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
# Using EX[:, None] for correct dimension alignment
plt.plot(t, np.mean(np.abs(X - EX[:, None]), axis=1), 'm', label='Sampled')

# Add legend, labels, and set y-axis limits
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel(r'E$|X-E(X)| = (2Var(X)/\pi)^{1/2}$')
plt.ylim([0, 0.02])

# Title of the plot, using LaTeX for mathematical symbols
plt.title('Ornstein-Uhlenbeck process: mean absolute deviation')


##########################################################################################################################################
#### 5B Autocovariance


# Initialize the autocovariance array
C = np.zeros((2 * nsteps + 1, npaths))

# Calculate the autocovariance for each path

for j in range(npaths):
    C[:, j] = correlate(X[:, j] - EX, X[:, j] - EX, 'full') / nsteps



# Average over all paths
C = C.mean(axis=1)

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
plt.plot(0, np.mean(np.var(X, axis=1, ddof=0)), 'bo', label='Average sampled Var')

# Add labels, legend, and title
plt.xlabel(r'$\tau$')
plt.ylabel(r'C($\tau$)')
plt.legend(loc='lower right')
plt.title('Ornstein-Uhlenbeck process: autocovariance')


##########################################################################################################################################
## 6B Autocorrelation

# The autocorrelation is the Covariance/Variance. However, since our OUP is
# only quasi-stationary (i.e. it is only stationary in the limit t -> inf)
# we will compute the autocorrelation as we have done above, in the limit
# as t -> inf

# It can be shown that in the limit, the autocorrelation becomes
# Corr(t,s) = exp(-1*alpha*tau)     with t < s

# Calculate theoretical autocorrelation
# For Ornstein-Uhlenbeck process, in the limit t -> inf, the autocorrelation becomes exp(-alpha * tau)
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
    timestep_index = int(time / dt)  # Convert time to the corresponding timestep index
    bin_counts, _ = np.histogram(X[timestep_index, :], bins=x, density=True)
    fs[:-1, i] = bin_counts

# Begin plotting
plt.figure(6, figsize=(10, 6))

# Plot analytical and sampled PDFs
for i, color in enumerate(colors):
    plt.plot(x, fa[:, i], color=color, linestyle='-', label=f'{t2[i]:.2f} (Analytical)')
    plt.plot(x, fs[:, i], color=color, linestyle='--', label=f'{t2[i]:.2f} (Sampled)')

# Create custom legend handles for line styles (Analytical and Sampled)
line_style_legend = [Line2D([0], [0], color='k', linestyle='-', label='Analytical'),
                     Line2D([0], [0], color='k', linestyle='--', label='Sampled')]

# Create and set the line style legend
plt.legend(handles=line_style_legend, loc='upper right', title="Line Type")

# Create custom legend handles for time points
time_legend = [Line2D([0], [0], color=colors[i], label=f'Time = {t2[i]:.2f}') for i in range(len(t2))]

# Create and set the time legend
plt.legend(handles=time_legend, loc='upper left', title="Time (t)", bbox_to_anchor=(1.05, 1))

# Add labels and title
plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.title('Ornstein-Uhlenbeck process: PDF at different times')

# Display the plot
plt.show()
