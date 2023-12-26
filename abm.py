import numpy as np
import matplotlib.pyplot as plt

# The formula for ABM is
# dX(t) = mu*dt + sigma*dW(t)

# Define parameters and time grid
npaths = 20000  # number of paths
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T/nsteps  # time step
t = np.linspace(0, T, nsteps+1)  # observation times
mu = -0.05  # model parameter
sigma = 0.4  # model parameter

# Monte Carlo
# Compute the increments with Euler-Maruyama
# Create an (npaths,nsteps) matrix to simulate the value at each time step
# along each path
dX = mu*dt + sigma*np.random.randn(npaths, nsteps)*np.sqrt(dt)

# Accumulate the increments
# Now we need to cumulatively sum the values over the time steps to get
# each path
X = np.hstack((np.zeros((npaths, 1)), np.cumsum(dX, axis=1)))
# stacks arrays in sequence horizontally (column-wise). In this context,
# it is taking the 2D array of zeros and appending the cumulative sum of dX to the right of it.
# Note the axis = 1 in cumsum to show we are adding each column to the prev. one

# Expected, mean and sample path
plt.figure(1)
EX = mu*t  # expected path
plt.plot(t, EX, 'k', label='Expected path')
plt.plot(t, np.mean(X, axis=0), ':k', label='Mean path')
plt.plot(t, X[::2000, :].T, alpha=0.3)
# plots every 2000th row i.e. row 2000, 4000 etc.
# since there are 20000 rows we have 10 lines on the plot
# transpose because Matplotlib's plot function expects the columns as
# individual series to plot against the x-axis but we have rows atm
plt.legend()
plt.xlabel('t')
plt.ylabel('X(t)')
plt.ylim([-1, 1])
plt.title('Arithmetic Brownian motion dX(t) = \mu dt + \sigma dW(t)')

# Variance = mean square deviation = mean square displacement of the random part
plt.figure(2)
plt.plot(t, sigma**2*t, label='Theory: \( \sigma^2t \)')
plt.plot(t, np.var(X, axis=0), label='Sampled')
plt.legend(loc='upper right')
plt.xlabel('t')
plt.ylabel('Var(X(t))')
plt.title('Arithmetic Brownian motion: Variance')

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

# Probability density function at different times
plt.figure(4)
time_indices = [20, 80, nsteps]  # times at which to plot the PDF - 20, 80, 200

for i, time_index in enumerate(time_indices, start=1):
    plt.subplot(3, 1, i)
    # 3 rows, 1 column, iterating through index i for 1st second and third plot)
    plt.hist(X[:, time_index], bins=np.arange(-1, 1.02, 0.02), density=True)
    plt.xlim([-1, 1])
    plt.ylim([0, 3.5])
    if i == 1:
        plt.ylabel('f_X(x,0.1)') # 20th step out of 200 is time 0.1
    elif i == 2:
        plt.ylabel('f_X(x,0.4)') # 80th step out of 200 is time 0.4
    else:
        plt.xlabel('x')
        plt.ylabel('f_X(x,1)') # 200th step out of 200 is time 1

plt.suptitle('Arithmetic Brownian motion: PDF at different times')

# Solution for Fokker-Planck
# Create a figure for plotting
fig = plt.figure(5)

# Calculate the diffusion coefficient from sigma
D = sigma**2 / 2  # diffusion coefficient

# Create a meshgrid for the space and time dimensions as the previous plot was on a new figure
x, tt = np.meshgrid(np.arange(-1, 1.02, 0.02), np.arange(0.1, 1.025, 0.025))

# Compute the solution of the Fokker-Planck equation again
f = 1/(2*np.sqrt(np.pi*D*tt))*np.exp(-(x-mu*tt)**2/(4*D*tt))

# Create a 3D axis for plotting using the corrected method
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, tt, f, cmap='viridis')

# Set the labels and title
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f_X(x, t)')
ax.set_title('Arithmetic Brownian motion: Solution of the Fokker-Planck equation')

# Set the view angle
ax.view_init(30, 24)

# Show the plot
plt.show()