# Importando librerias necesarias
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

def figuras1(mejor_par, n_iter, F):
    
    fig1 = plt.figure()
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