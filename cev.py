import numpy as np
import matplotlib.pyplot as plt

class ProcessCEV:

    def __init__(self, mu, sigma, gamma):
        self._mu = mu
        self._sigma = sigma
        self._gamma = gamma

    def Simulate(self, T=1, dt=0.001, S0=1.):
        n = round(T / dt)

        mu = self._mu
        sigma = self._sigma
        gamma = self._gamma

        gaussian_increments = np.random.normal(size=n - 1)
        res = np.zeros(n)
        res[0] = S0
        S = S0
        sqrt_dt = dt ** 0.5
        for i in range(n - 1):
            S = S + S * mu * dt + sigma * \
                (S ** gamma) * gaussian_increments[i] * sqrt_dt
            res[i + 1] = S

        return res

T = 20
dt = 0.01
plt.plot(ProcessCEV(0.05, 0.15, 0.5).Simulate(
    T, dt), label="CEV (gamma = 0.5)")
plt.plot(ProcessCEV(0.05, 0.15, 1.5).Simulate(
    T, dt), label="CEV (gamma = 1.5)")

plt.xlabel('Time index')
plt.ylabel('Value')

plt.title("Realization of CEV")

plt.legend()

plt.show()