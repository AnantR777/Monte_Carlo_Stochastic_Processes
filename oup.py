import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

#  The formula for OUP is
#  dX(t) = alpha*(mu - X)*dt + sigma*dW(t)
# This could also be described as the Vasicek model, however, it is worth noting
# they are the same thing. Vasicek just applied OUP to finance,
# specifically interest rates.

# Define the parameters and the time grid
npaths = 20000  # number of paths
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.arange(0, T + dt, dt)  # observation times
alpha, mu, sigma = 5, 0.07, 0.07  # model parameters
X0 = 0.03  # initial value

# Monte Carlo
# Allocate and initialize all paths
X = np.zeros((nsteps + 1, npaths))
X[0, :] = X0

# Sample standard Gaussian random numbers
N = np.random.randn(nsteps, npaths)

# ----------------------------------------------
# 1. Euler-Maruyama Method
# ----------------------------------------------

# for i from 1 to nsteps
#     X(:,i+1) = X(:,i) + alpha*(mu-X(:,i))*dt + sigma*sqrt(dt)*N(:,i) ;
# end

# ----------------------------------------------
# 2. Euler-Maruyama Method with Analytic Moments
# ----------------------------------------------
# To use the analytic moments method we need analytic expressions for the
# expectation E(X) and varaince Var(X)
# For the OUP we have these expressions (see Ballotta & Fusai p.94)

# E(X) = X0*exp(-alpha*t) + mu*( 1-exp(-alpha*t) )
# Var(X) = (sigma^2/2*alpha) * ( 1-exp(-2*alpha*t) )

# We then ignore the form of our model and compute:
# dX = E(X) + sqrt(Var(X))*randn()
# Substituting our dt for t, and X0 with the X from the previous timestep

# Compute the standard deviation for a time step
# sdev = sigma*np.sqrt(dt) # plain Euler-Maruyama

start_time_oup = time.perf_counter()
sdev = np.sqrt((sigma ** 2) / (2 * alpha) * (1 - np.exp(
    -2 * alpha * dt)))  # Euler-Maruyama with analytic moments

# Compute and accumulate the increments
for i in range(nsteps):
    # X[i+1,:] = X[i,:] + alpha*(mu-X[i,:])*dt + sdev*N[i,:] # plain Euler-Maruyama
    X[i + 1, :] = mu + (X[i, :] - mu) * np.exp(-alpha * dt) + sdev * N[i,
                                                                     :]  # Euler-M. with a. m.
end_time_oup = time.perf_counter()
# Mean:____
# Expected, mean and sample paths, long-term average
# expected path
EX = mu + (X0 - mu) * np.exp(-alpha * t)
# sample mean:
Sampledmean = np.mean(X, axis=1)
plt.plot(t, EX, 'k')
plt.plot(t, Sampledmean, 'r:')
plt.plot(t, mu * np.ones_like(t), 'b--')
plt.plot(t, X[:, ::1000], alpha=0.4)
plt.legend(['Expected path', 'Mean path', 'Long-term average'])
plt.xlabel('t')
plt.ylabel('X')

sdevinfty = sigma / np.sqrt(2 * alpha)
plt.ylim([mu - 4 * sdevinfty, mu + 4 * sdevinfty])
plt.title(r'Ornstein-Uhlenbeck process $dX = \alpha(\mu - X)dt + \sigma dW$')

# Variance:____
plt.figure(2)
theoryVar = sigma ** 2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
SampledVar = np.var(X, axis=1)
plt.plot(t, SampledVar, 'm')
plt.plot(t, theoryVar, 'r')
plt.plot(t, sigma ** 2 * t, 'g')
plt.plot(t, sigma ** 2 / (2 * alpha) * np.ones(t.size), 'b')
plt.plot(t, np.mean((X.T - EX) ** 2, axis=0), 'c--', label='Sampled 2')

plt.legend(['Theory', 'Sampled 1', r'$\sigma^2 t$', r'$\sigma^2/(2\alpha)$',
            'Sampled 2'], loc='lower right')
plt.xlabel('t')
plt.ylabel('Var(X) = E((X-E(X))^2)')
plt.ylim([0, 0.0006])
plt.title('Ornstein-Uhlenbeck process: variance')

# Mean absolute deviation_______
plt.figure(3)
plt.plot(t, sigma * np.sqrt((1 - np.exp(-2 * alpha * t)) / (np.pi * alpha)),
         'r', label='Theory')
plt.plot(t, sigma * np.sqrt(2 * t / np.pi), 'g',
         label=r'$\sigma(2t/\pi)^{1/2}$')
plt.plot(t, sigma / np.sqrt(np.pi * alpha) * np.ones(t.shape), 'b',
         label='Long-term average')
plt.plot(t, np.mean(np.abs(X.T - EX), axis=0), 'm', label='Sampled')
plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel(r'E$(|X-E(X)|)$ = (2Var(X)/$\pi)^{1/2}$')
plt.ylim([0, 0.02])
plt.title('Ornstein-Uhlenbeck process: mean absolute deviation')

# PDF @ different times _________
# Probability density function at different times
x = np.linspace(-0.02, mu + 4 * sdevinfty, 200)
t2 = np.array([0.05, 0.1, 0.2, 0.4, 1])
EX2 = mu + (X0 - mu) * np.exp(-alpha * t2)
sdev = sigma * np.sqrt((1 - np.exp(-2 * alpha * t2)) / (2 * alpha))
fa = np.zeros((len(x), len(t2)))  # analytical
fs = np.zeros((len(x) - 1, len(t2)))  # sampled

for i in range(len(t2)):
    fa[:, i] = norm.pdf(x, loc=EX2[i], scale=sdev[i])
    hist, bin_edges = np.histogram(X[int(t2[i] * nsteps), :], bins=x,
                                   density=True)
    fs[:, i] = hist

plt.figure(4)
plt.plot(x, fa)
plt.plot(x[:-1], fs)
plt.legend(['t = 0.05', 't = 0.10', 't = 0.20', 't = 0.40', 't = 1.00'])
plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.title('Ornstein-Uhlenbeck process: PDF at different times')

# Autocovariance:___________
EX = mu + (X0 - mu) * np.exp(-alpha * t)
C = np.zeros((2 * nsteps + 1, npaths))
for j in range(npaths):
    C[:, j] = np.correlate(X[:, j] - EX, X[:, j] - EX, mode='full') / (
                nsteps + 1)  # unbiased estimator
# sampled Cov:
C = np.mean(C, axis=1)
theoryAutoCov = sigma ** 2 / (2 * alpha) * np.exp(-alpha * t)
fig, ax = plt.subplots()
ax.plot(t, theoryAutoCov, 'r')
ax.plot(t, C[nsteps:], 'g')
# With t=0
ax.plot(0, sigma ** 2 / (2 * alpha), 'go')
ax.plot(0, np.mean(np.var(X, axis=1)), 'bo')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$C(\tau)$')

ax.legend(
    ['Theory for infinite t', 'Sampled', 'Var for infinite t', 'Sampled Var'])
ax.set_title('Ornstein-Uhlenbeck process: autocovariance')

# Autocorrelation:_____________
# The autocorrelation is the Covariance/Variance. However, since our OUP is
# only quasi-stationary (i.e. it is only stationary in the limit t -> inf)
# we will compute the autocorrelation as we have done above, in the limit
# as t -> inf

# It can be shown that in the limit, the autocorrelation becomes
# Corr(t,s) = exp(-1*alpha*tau)     with t < s

fig, ax = plt.subplots()
theoryAutoco = np.exp(-alpha * t)
sampledAutoco = C[nsteps:] / C[nsteps]
ax.plot(t, theoryAutoco, 'r')
ax.plot(t, sampledAutoco, 'g')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$c(\tau)$')
ax.legend(['Theory for infinite t', 'Sampled'])
ax.set_title('Ornstein-Uhlenbeck process: autocorrelation')
plt.show()

print(end_time_oup - start_time_oup)