import numpy as np
from scipy.stats import ncx2, norm
import matplotlib.pyplot as plt
import time

#  The formula for the FSRP is
#  dX(t) = alpha*(mu - X)*dt + sigma*sqrt(X)*dW(t)
# As with the OUP/Vasicek Model, the FSRP goes by another name in Finance
# and that's the Cox-Ingersoll-Ross Process (CIRP). Again they applied the
# process to model interest rates.

# Define parameters and time grid
npaths = 2000  # number of paths
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.linspace(0, T, nsteps + 1)  # observation times
alpha = 5
mu = 0.07
sigma = 0.265  # model parameters
# alpha = 5
# mu = 0.03
# sigma = 0.8 # model parameters

X0 = 0.03  # initial value
# We introduce a variable for monitoring purposes. If our feller ratio,
# defined below, is > 1, then X will never reach 0.
Feller_ratio = 2 * alpha * mu / sigma ** 2  # for monitoring
# Monte Carlo

# Allocate and initialise all paths
X = np.zeros((nsteps + 1, npaths))
X[0] = X0

start_time_fsrp = time.perf_counter()
# Euler-Maruyama
N = np.random.randn(nsteps, npaths)  # sample standard normal random numbers
a = sigma ** 2 / alpha * (np.exp(-alpha * dt) - np.exp(
    -2 * alpha * dt))  # with analytic moments
b = mu * sigma ** 2 / (2 * alpha) * (
            1 - np.exp(-alpha * dt)) ** 2  # with analytic moments
for i in range(nsteps):  # compute and accumulate the increments
    # X[i+1,:] = X[i,:] + alpha*(mu-X[i,:])*dt + sigma*np.sqrt(X[i,:]*dt)*N[i,:] # plain
    X[i + 1, :] = mu + (X[i, :] - mu) * np.exp(-alpha * dt) + np.sqrt(
        a * X[i, :] + b) * N[i, :]  # with analytic moments
    # if X[i+1,:] < 0:
    #   X[i+1,:] = 0
    # X[i+1,:] = X[i+1,:]*(X[i+1,:]>=0)
    # makes sure that x>0
    X[i + 1, :] = np.maximum(X[i + 1, :], np.zeros(npaths))

# Exact method
# d = 4*alpha*mu/sigma**2 # degrees of freedom of the non-central chi square distribution
# k = sigma**2*(1-np.exp(-alpha*dt))/(4*alpha)
# for i in range(nsteps): # compute and accumulate the increments
#    lambda = 4*alpha*X[i,:]/(sigma**2*(np.exp(alpha*dt)-1))
#   #X[i+1,:] = ncx2.ppf(np.random.rand(npaths),d,lambda)*k; i % 80000 times slower than EM
#    X[i+1,:] = ncx2.rvs(d,lambda, size=npaths)*k # 40 times slower than EM
# end
end_time_fsrp = time.perf_counter()

# mean:__________
# Expected, mean and sample paths
EX = mu + (X0 - mu) * np.exp(-alpha * t)
plt.figure(1)
plt.plot(t, EX, 'k')
plt.plot(t, np.mean(X, axis=1), ':k')
plt.plot(t, mu * np.ones_like(t), 'k--')
plt.plot(t, X[:, ::1000], alpha=0.4)
# plt.plot(t, np.mean(X, axis=1), ':k')
plt.legend(['Expected path', 'Mean path', r'$\mu$'])
plt.xlabel('t')
plt.ylabel('X')

sdevinfty = sigma * np.sqrt(mu / (2 * alpha))
plt.ylim([-0.02, mu + 4 * sdevinfty])
plt.title(
    'Paths of a Feller square-root process dX = \u03B1(\u03BC-X)dt + \u03C3X^{1/2}dW')

# Probability density function at different times
t2 = np.array([0.05, 0.1, 0.2, 0.4, 1])
x = np.linspace(-0.02, mu+4*sdevinfty, 200)
k = sigma**2*(1-np.exp(-alpha*t2))/(4*alpha)
d = 4*alpha*mu/sigma**2
lambda_ = 4*alpha*X0/(sigma**2*(np.exp(alpha*t2)-1)) # non-centrality parameter
fa = np.zeros((len(x),len(t2))) # analytical
fs = np.zeros((len(x)-1,len(t2))) # sampled
for i in range(len(t2)):
    fa[:,i] = ncx2.pdf(x/k[i], d, lambda_[i])/k[i]
    fs[:,i], _ = np.histogram(X[int(t2[i]/dt),:], bins=x, density=True)
    #fs[:,i], _ = np.histogram(X[int(t2[i]*nsteps),:], bins=x, density=True)
plt.figure(2)
plt.plot(x, fa, x[:-1], fs)
plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.legend(['t = 0.05', 't = 0.10', 't = 0.20', 't = 0.40', 't = 1.00'])
plt.title('Probability density function of a Feller square-root process at different times')
plt.show()

print(end_time_fsrp - start_time_fsrp)