import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y)

# Set the x and y axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Sin(x)')

# Show the plot
plt.show()

# Define the range of x values
#x_min, x_max = -8, 8
#y_min, y_max = -8, 8
#n_points = 100
#x_values = np.linspace(x_min, x_max, n_points)
#y_values = np.linspace(y_min, y_max, n_points)

# Create a meshgrid of x and y values
#X, Y = np.meshgrid(x_values, y_values)

# Evaluate the function at each point in the meshgrid
#Z = F1([X, Y])

# Create a 2D contour plot of the function
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].contourf(X, Y, Z, levels=30, cmap='cool')
#ax[0].set_xlabel('x1')
#ax[0].set_ylabel('x2')
#ax[0].set_title('Contour plot of F1(x)')

# Create a 3D surface plot of the function
#ax[1] = fig.add_subplot(122, projection='3d')
#ax[1].plot_surface(X, Y, Z, cmap='cool')
#ax[1].set_xlabel('x1')
#ax[1].set_ylabel('x2')
#ax[1].set_zlabel('F1(x)')
#ax[1].set_title('3D surface plot of F1(x)')

# Show the plot
#plt.show()
