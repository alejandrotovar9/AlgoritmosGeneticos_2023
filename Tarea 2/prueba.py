import numpy as np
import matplotlib.pyplot as plt

def plot_function(F, points=None):
    # Define the domain of the function
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the meshgrid
    Z = F([X, Y])

    # Extract the scattered points from the points argument
    if points is not None:
        points = np.array(points)
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = F([x_points, y_points])

    # Create a 3D surface plot of the function
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)

    # Add the scattered points to the plot
    if points is not None:
        ax.scatter(x_points, y_points, z_points, color='r')

    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('My Function with Scattered Points' if points is not None else 'My Function')

    plt.show()

def F1(x):
    return x[0]**2 + 2*(x[1]**2) - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7

def F2(x):
    return 0.5 - (np.sin(np.sqrt(x[0]**2+x[1]**2))**2 - 0.5)/(1 + 0.001*(x[0]**2 + x[1]**2))**2

# Define the array of points to scatter on the mesh
points = [[0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5]]

# Call the function to create the plot for F1 with scattered points
plot_function(F1, points)

# Call the function to create the plot for F2 with scattered points
plot_function(F2, points)
