import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.signal import correlate
from matplotlib.lines import Line2D

# Timesteps as ROWS!
# Paths as COLUMNS!
#   path1    path2
# t0 ( 0  ,  0  ... )
# t1 (0.1 , -0.3 .. )
# t2 (0.4 , 0.1 ... )

# Simulation of a Feller Square Root Process using Monte Carlo Simulation
# ---------------------------------------------------------------------
# A Feller Square Root Process is a stochastic model used in finance to describe
# the volatility of certain assets. It is often used in options pricing and risk
# management.

# Parameters:
# T = Total time duration
# nsteps = Number of steps in the simulation
# npaths = Number of independent paths to simulate
# start_point = Starting point of the process
# alpha = Mean reversion rate
# sigma = Volatility
# mu = Long-term mean

# The formula for a Feller Square Root Process is:
# dX(t) = alpha * (mu - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)

# The key feature of this process is that it exhibits mean reversion, causing
# the process to move towards its long-term mean over time. The volatility term
# is proportional to the square root of the current value of the process.

# Simulating a Feller Square Root Process involves generating random paths that
# adhere to this stochastic differential equation.

# This process is commonly used in financial modeling, particularly in modeling
# interest rates and volatility.

# As with the OUP/Vasicek Model, the FSRP goes by another name in Finance
# and that's the Cox-Ingersoll-Ross Process (CIRP). Again they applied the
# process to model interest rates.

# Parameters
npaths = 20000  # Number of simulations
T = 1  # Time horizon
nsteps = 200  # Number of timesteps
dt = T / nsteps  # Size of the timesteps
t = np.linspace(0, T, nsteps + 1)  # Discretization of the grid
alpha = 5
mu = 0.07
sigma = 0.265
X0 = 0.03  # Initial value

# initialisation
# We introduce a variable for monitoring purposes. If our feller ratio,
# defined below, is > 1, then X will never reach 0.
feller_ratio = (2*alpha*mu)/(sigma**2)

# Since we now have our variable X with the equation (as part of the dt &
# dW terms) we can no longer simply compute an entre matrix for dX. We must
# use an iterative approach.

# Set up an [nsteps,npaths] matrix, with the first column all equal to X0
# (i.e. all paths start at X0) and the rest zeros (these will be filled out
# later).
X = np.zeros((nsteps + 1, npaths))
X[0, :] = X0


# Define an [nsteps,npaths] matrix of normally distributed random numbers
N = np.random.randn(nsteps, npaths)

### Simulation methods

# ----------------------------------------------
# 1. Euler-Maruyama Method
# ----------------------------------------------

for i in range(nsteps):
    X[i + 1, :] = X[i, :] + alpha * (mu - X[i, :]) * dt + sigma * np.sqrt(X[i, :] * dt) * N[i, :]
    X[i + 1, :] = np.maximum(X[i + 1, :], 0)  # Ensuring non-negative values

# ----------------------------------------------
# 2. Euler-Maruyama Method with Analytic Moments
# ----------------------------------------------

# To use the analytic moments method we need analytic expressions for the
# expectation E(X) and varaince Var(X)
# For the OUP we have these expressions (see Ballotta & Fusai p.94)

# E(X) = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) )
# Var(X) = X0*(sigma^2/alpha)*(exp(-1*alpha*t) - exp(-2*alpha*t) ) + mu*(sigma^2/2*alpha)*(1-exp(-1*alpha*t))^2

# We then ignore the form of our model and compute:
# dX = E(X) + sqrt(Var(X))*randn()
# Substituting our dt for t, and X0 with the X from the previous timestep

# Since Var(X) is long and cumbersome, we will break it up into two pieces
# This will allow us to write the Var(X) = aX+b

# Coefficients for the variance equation
a = (sigma**2 / alpha) * (np.exp(-alpha * dt) - np.exp(-2 * alpha * dt))
b = mu * (sigma**2 / (2 * alpha)) * (1 - np.exp(-alpha * dt))**2



for i in range(nsteps):
    EX = X[i, :] * np.exp(-alpha * dt) + mu * (1 - np.exp(-alpha * dt))
    VarX = a * X[i, :] + b
    X[i + 1, :] = EX + np.sqrt(VarX) * N[i, :]
    X[i + 1, :] = np.maximum(X[i + 1, :], 0)

# ----------------------------------------------
# 3. Exact Method
# ----------------------------------------------
# It turns out we can actually compute the simulation exactly by using the
# non-central Chi-sq. distribution. We need to calculate some further
# parameters first though.
# Recall our N.C. Chi-sq. Distribution needs:
# d : degrees of freedom
# lambda : non-centrality parameter (which will change for each loop) see
# Ballotta & Fusai p.111
# We also need k (a multiplying factor)


# Exact Method parameters
d = 4 * alpha * mu / sigma ** 2
k = sigma ** 2 * (1 - np.exp(-alpha * dt)) / (4 * alpha)

for i in range(nsteps):
    lambda_param = 4 * alpha * X[i, :] / (sigma ** 2 * (np.exp(alpha * dt) - 1))
    X[i + 1, :] = ncx2.rvs(df=d, nc=lambda_param, size=npaths) * k

# Calculate the expected path
EX = X0 * np.exp(-alpha * t) + mu * (1 - np.exp(-alpha * t))

# Calculate the long-term standard deviation
sdev_infty = sigma * np.sqrt(mu / (2 * alpha))

# Plotting
plt.figure(1)
plt.plot(t, EX, 'r', label='Expected path')
plt.plot(t, np.mean(X, axis=1), ':k', label='Mean path')
plt.plot(t, mu * np.ones_like(t), 'k--', label='$\mu$')
for i in range(0, npaths, 1000):
    plt.plot(t, X[:, i], alpha=0.5)

# Setting plot properties
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('X')
plt.ylim([-0.02, mu + 4 * sdev_infty])
plt.title('Paths of a Feller Square-Root Process $dX = \\alpha(\mu-X)dt + \\sigma X^{1/2}dW$')

# Theoretical variance of X
VARX = X0 * (sigma**2 / alpha) * (np.exp(-alpha * t) - np.exp(-2 * alpha * t)) \
       + mu * (sigma**2 / (2 * alpha)) * (1 - np.exp(-alpha * t))**2

# Asymptote for variance as t -> 0+
varzero = X0 * sigma**2 * t

# Asymptote for variance as t -> inf
varinf = mu * (sigma**2 / (2 * alpha)) * np.ones_like(t)

# Sample variance
sampled_variance = np.var(X, axis=1)

# Mean square deviation
mean_square_deviation = np.mean((X - EX[:, np.newaxis])**2, axis=1)

# Plotting
plt.figure(2)
plt.plot(t, VARX, 'r', label='Theory')
plt.plot(t, varzero, 'g', label='$X_0\sigma^2t$')
plt.plot(t, varinf, 'b', label='$\mu\sigma^2/(2\\alpha)$')
plt.plot(t, sampled_variance, 'm', label='Sampled 1')
plt.plot(t, mean_square_deviation, 'c--', label='Sampled 2')
plt.legend(loc='lower right')
plt.xlabel('Time (t)')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.ylim([0, 0.0006])
plt.title('Feller Square-Root Process: Variance')

# Asymptote for variance as t -> inf
varinf_pt = mu * (sigma**2 / (2 * alpha))

# Initialize the autocovariance array
C = np.zeros((npaths, 2 * nsteps + 1))

# Calculate the autocovariance for each path
for i in range(npaths):
    deviation = X[:, i] - EX  # Adjust deviation calculation
    # Compute autocorrelation and normalize by the number of steps
    C[i, :] = correlate(deviation, deviation, mode='full') / nsteps

# Average the autocovariance across all paths
C_mean = np.mean(C, axis=0)

# Theoretical autocovariance as t -> inf
COVX = np.exp(-alpha * t) * (sigma**2 * mu) / (2 * alpha)

# Plotting
plt.figure(3)
plt.plot(t, COVX, 'r', label='Theory')
plt.plot(t, C_mean[nsteps:], 'b', label='Sampled')
plt.plot(0, varinf_pt, 'ro', label='Var for infinite t')
plt.plot(0, np.mean(np.var(X, axis=1)), 'bo', label='Average sampled Var')  # Adjust axis for var calculation
plt.xlabel('Lag ($\\tau$)')
plt.ylabel('C($\\tau$)')
plt.legend(loc='upper right')
plt.title('Feller Square-Root Process: Autocovariance')

## Autocorrelation

# Taking COVX/Var(X) in the limit t -> inf, it can be shown that the
# autocorrelation becomes
# Corr(s,t) = exp(-1*alpha*tau)     with t < s
# Theoretical autocorrelation
# Theoretical autocorrelation
CORRX = np.exp(-alpha * t)

# Normalize the averaged autocovariance to get the autocorrelation
# Since C_mean is already calculated, we use it directly
sampled_autocorr = C_mean / C_mean[nsteps]

# Adjust the time array for plotting
# We need to consider only half of the time array since we are looking at lags
t_lag = t[:len(sampled_autocorr) - nsteps]

# Plotting the autocorrelation
plt.figure(4)
plt.plot(t_lag, CORRX[:len(t_lag)], 'r', label='Theory')  # Theoretical autocorrelation
plt.plot(t_lag, sampled_autocorr[nsteps:], 'b', label='Sampled')  # Sampled autocorrelation
plt.xlabel('Lag ($\\tau$)')
plt.ylabel('c($\\tau$)')
plt.legend()
plt.title('Feller Square-Root Process: Autocorrelation')

# Calculate the standard deviation in the limit as t -> infinity
sdevinfty = sigma * np.sqrt(mu / (2 * alpha))

# Time points and range for x
t2 = np.array([0.05, 0.1, 0.2, 0.4, 1])
x = np.linspace(-0.02, mu + 4 * sdevinfty, 200)

# Parameters for the non-central chi-square distribution
k = sigma**2 * (1 - np.exp(-alpha * t2)) / (4 * alpha)
d = 4 * alpha * mu / sigma**2
lambd = 4 * alpha * X0 / (sigma**2 * (np.exp(alpha * t2) - 1))

# Initialize arrays for analytical and sampled PDFs
fa = np.zeros((len(x), len(t2)))  # Analytical
fs = np.zeros((len(x), len(t2)))  # Sampled

# Define colors
colors = ['red', 'green', 'blue', 'orange', 'black']

# Compute the PDFs and plot
plt.figure(5)
for i, current_time in enumerate(t2):
    color = colors[i]
    # Calculate index in the X matrix corresponding to the time point t
    time_index = int(current_time / T * nsteps)
    fa[:, i] = ncx2.pdf(x / k[i], d, lambd[i]) / k[i]
    bin_counts, _ = np.histogram(X[time_index, :], bins=x, density=True)
    fs[:-1, i] = bin_counts

    plt.plot(x, fa[:, i], label=f't = {current_time:.2f} (Analytical)', color=color)
    plt.plot(x, fs[:, i], label=f't = {current_time:.2f} (Sampled)', linestyle='--', color=color)

# Create custom legend handles for time points
color_legend_handles = [Line2D([0], [0], color=colors[i], label=f'Time = {t2[i]:.2f}') for i in range(len(t2))]

# Add handles for "Analytical" and "Sampled" line types
color_legend_handles.append(Line2D([0], [0], color='k', linestyle='-', label='Analytical'))
color_legend_handles.append(Line2D([0], [0], color='k', linestyle='--', label='Sampled'))

# Create and set the combined legend
plt.legend(handles=color_legend_handles, loc='upper right')

plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.title('Probability density function of a Feller square-root process at different times')

plt.show()


