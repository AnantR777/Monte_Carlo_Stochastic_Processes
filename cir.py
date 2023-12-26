import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ncx2


# Define parameters and time grid
npaths = 20000  # number of paths
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.linspace(0, T, nsteps+1)  # observation times
alpha = 5
mu = 0.07
sigma = 0.265
X0 = 0.03

# Allocate and initialize all paths
X = np.zeros((nsteps+1, npaths))
X[0, :] = X0

# Sample standard Gaussian random numbers
N = np.random.randn(nsteps, npaths)

# Compute and accumulate the increments
a = sigma**2 / alpha * (np.exp(-alpha*dt) - np.exp(-2*alpha*dt))  # Euler with analytic moments
b = mu*sigma**2 / (2*alpha) * (1 - np.exp(-alpha*dt))**2  # Euler with analytic moments
for i in range(nsteps):
    X[i+1, :] = mu + (X[i, :] - mu) * np.exp(-alpha*dt) + np.sqrt(a*X[i, :] + b) * N[i, :]  # Euler with a.m.

# Mean________
EX = mu + (X0 - mu) * np.exp(-alpha*t)

# Plot the expected, mean and sample paths
plt.figure(1)
plt.plot(t, EX, 'k', t, np.mean(X, axis=1), ':k', t, X[:, ::1000],alpha=0.4)
plt.legend(['Expected path', 'Mean path'])
plt.xlabel('t')
plt.ylabel('X')
sdevinfty = sigma * np.sqrt(mu / (2*alpha))
plt.ylim([0, mu + 4*sdevinfty])
plt.title('Paths of a Cox-Ingersoll-Ross process dX = alpha(mu - X)dt + sigma*sqrt(X)*dW')


# Variance__________
plt.figure(2)
variance = X0 * sigma**2 / alpha * (np.exp(-alpha*t) - np.exp(-2*alpha*t)) + mu*sigma**2 / (2*alpha) * (1 - np.exp(-alpha*t))**2
var0 = X0 * sigma**2 * t
varinfty = mu*sigma**2 / (2*alpha) * np.ones(t.shape)

plt.plot(t, variance, 'r')
plt.plot(t, var0,'g',)
plt.plot( t, varinfty, 'b')
plt.plot( t, np.var(X, axis=1), 'm')
plt.legend(['Theory', 'X_0*sigma^2*t', 'mu*sigma^2/(2*alpha)', 'Sampled'], loc='upper left')
plt.xlabel('t')
plt.ylabel('Var(X)')
plt.ylim([0, 0.0006])
plt.title('Variance of a Cox-Ingersoll-Ross process dX = alpha(mu - X)dt + sigma*sqrt(X)*dW')

plt.figure(3)
# Compute and plot the probability density function at different times
t2 = np.array([0.05, 0.1, 0.2, 0.4, 1])
x = np.linspace(0, mu+4*sdevinfty, 200)
k = sigma**2*(1-np.exp(-alpha*t2))/(4*alpha)
d = 4*alpha*mu/sigma**2
lmbda = 4*alpha*X0/(sigma**2*(np.exp(alpha*t2)-1)) # non-centrality parameter
f = np.zeros((len(x), len(t2)))
for i in range(len(t2)):
    f[:,i] = ncx2.pdf(x/k[i], d, lmbda[i])/k[i]

plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.legend(['t = 0.05', 't = 0.10', 't = 0.20', 't = 0.40', 't = 1.00'])
plt.title('Probability density function of a Cox-Ingersoll-Ross process at different times')
plt.savefig('cirpdensities.pdf')
plt.show()