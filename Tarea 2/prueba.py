import numpy as np
import matplotlib.pyplot as plt

# Define the function to plot
def my_function(x, y):
    return np.sin(x) + np.cos(y)

# Define the domain of the function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Evaluate the function on the meshgrid
Z = my_function(X, Y)

# Generate some random scattered points
num_points = 100
x_points = np.random.uniform(-5, 5, num_points)
y_points = np.random.uniform(-5, 5, num_points)
z_points = my_function(x_points, y_points)

# Create a 3D surface plot of the function
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)

# Add the scattered points to the plot
ax.scatter(x_points, y_points, z_points, color='r')

# Set the axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('My Function with Scattered Points')

plt.show()