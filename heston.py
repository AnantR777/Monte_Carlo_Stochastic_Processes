import numpy as np
import matplotlib.pyplot as plt

# dS = mu*S*dt + sqrt(V)*S*dW1 <=> dX = (mu-V/2)*dt + sqrt(V)*dW1
# dV = kappa*(Vbar-V)*dt + sigmaV*sqrt(V)*dW2
# dW1*dW2 = rho*dt
# The Black-Scholes-Merton price is given for reference
# dS = mu*S*dt + sigma*S*dW

# Time grid and Monte Carlo parameters
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.arange(0, T + dt, dt)  # observation times
npaths = 20000  # number of paths

# Market parameters
S0 = 1  # initial stock price
r = 0.02  # risk-free interest rate
q = 0  # dividend rate

# Model parameters
kappa = 3  # mean-reversion rate
Vbar = 0.1  # mean-reversion level
sigmaV = 0.25  # volatility of the volatility (vol of vol)
V0 = 0.08  # initial volatility
rho = -0.8  # correlation of W1 and W2

mu = r - q  # GBM drift parameter
Feller_ratio = 2 * kappa * Vbar / sigmaV**2  # for monitoring

# Monte Carlo Simulation
np.random.seed(0)  # for reproducibility
N1 = np.random.randn(nsteps, npaths)
N2 = rho * N1 + np.sqrt(1 - rho**2) * np.random.randn(nsteps, npaths)

# Allocate and initialise the paths for the variance and the log price
V = np.vstack([V0 * np.ones(npaths), np.zeros((nsteps, npaths))])

# Euler-Maruyama for the variance
a = sigmaV**2 / kappa * (np.exp(-kappa * dt) - np.exp(-2 * kappa * dt))
b = Vbar * sigmaV**2 / (2 * kappa) * (1 - np.exp(-kappa * dt))**2
for i in range(nsteps):
    V[i + 1, :] = Vbar + (V[i, :] - Vbar) * np.exp(-kappa * dt) + \
        np.sqrt(a * V[i, :] + b) * N2[i, :]
    V[i + 1, :] = np.maximum(V[i + 1, :], 0)  # avoid negative V

# Compute the increments of the arithmetic Brownian motion X = log(S/S0)
dX = (mu - 0.5 * V[:-1, :]) * dt + np.sqrt(V[:-1, :]) * N1 * np.sqrt(dt)

# Accumulate the increments
X = np.vstack([np.zeros(npaths), np.cumsum(dX, axis=0)])

# Transform to geometric Brownian motion
S = S0 * np.exp(X)

# Plotting the results
plt.figure(1)
ES = S0 * np.exp(mu * t)  # expected path
plt.plot(t, ES, 'k', label='Expected path')
plt.plot(t, np.mean(S, axis=1), ':k', label='Mean path')
plt.plot(t, S[:, ::1000])
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([0, 2.5])
plt.title('Geometric Brownian motion dS = \muSdt + \sigmaSdW sample paths')
plt.legend()

# Probability density function at different times
plt.figure(2)

plt.subplot(3, 1, 1)
plt.hist(S[20, :], bins=np.arange(0, 2.5, 0.025), density=True)
plt.ylabel('f_S(x,0.1)')
plt.xlim([0, 2.5])
plt.ylim([0, 5])
plt.title('Heston model: PDF of S at different times')

plt.subplot(3, 1, 2)
plt.hist(S[80, :], bins=np.arange(0, 2.5, 0.025), density=True)
plt.ylabel('f_S(x,0.4)')
plt.xlim([0, 2.5])
plt.ylim([0, 5])

plt.subplot(3, 1, 3)
plt.hist(S[-1, :], bins=np.arange(0, 2.5, 0.025), density=True)
plt.xlabel('x')
plt.ylabel('f_S(x,1.0)')
plt.xlim([0, 2.5])
plt.ylim([0, 5])

plt.show()