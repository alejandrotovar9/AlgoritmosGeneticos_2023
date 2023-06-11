# Importando librerias necesarias
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

def figuras1(mejor_par, n_iter, F):

    mejor_par = np.array(mejor_par)
    # Evaluate the F function for each pair
    W = np.zeros(n_iter)
    for i in range(n_iter):
        W[i] = F(mejor_par[i])
    # Create a scatter plot of the x-y value pairs with the W values as color
    plt.scatter(mejor_par[:, 0], mejor_par[:, 1], c=W, s=20)
    #identificando los puntos
    # Add text labels to the plot
    for i in range(n_iter):
        plt.text(mejor_par[i, 0], mejor_par[i, 1], i+1, ha='center', va='center')
    # Add colorbar and labels to the plot
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ubicacion de los mejores individuos ubicados en el plano 2D')
    plt.show()

def figuras2(generaciones, mejores, prom):

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first figure on the first subplot
    ax1.plot(generaciones, mejores)
    ax1.set_xlabel('Generaciones')
    ax1.set_ylabel('Fitness del mejor individuo')
    ax1.set_title('Evolución del Algoritmo Genético')

    # Plot the third figure on the second subplot
    ax2.plot(generaciones, prom)
    ax2.set_xlabel('Generaciones')
    ax2.set_ylabel('Fitness promedio de los individuos')
    ax2.set_title('Evolución del Algoritmo Genético')

    # Show the plots
    plt.show()

def figuras3(F, points=None, dom=None):
    if dom is not None:
        x = np.linspace(dom[0][0], dom[0][1], 100)
        y = np.linspace(dom[1][0], dom[1][1], 100)
    if dom == [[-100.0,100.0],[-100.0,100.0]]:
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
    else:
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