# Importando librerias necesarias
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
#Scatter
def figuras1(mejor_par, n_iter, F):

    mejor_par = np.array(mejor_par)
    # Evaluar la funcion F para cada par
    W = np.zeros(n_iter)
    for i in range(n_iter):
        W[i] = F(mejor_par[i])
    # Crear scatter plot con pares x-y
    plt.scatter(mejor_par[:, 0], mejor_par[:, 1], c=W, s=20)
    #identificando los puntos
    for i in range(n_iter):
        plt.text(mejor_par[i, 0], mejor_par[i, 1], i+1, ha='center', va='center')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ubicacion de los mejores individuos ubicados en el plano 2D')
    plt.show()

#Promedio y Fitness Maximo
def figuras2(generaciones, mejores, prom):

    #Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(generaciones, mejores)
    ax1.set_xlabel('Generaciones')
    ax1.set_ylabel('Fitness del mejor individuo')
    ax1.set_title('Evolución del Algoritmo Genético')

    ax2.plot(generaciones, prom)
    ax2.set_xlabel('Generaciones')
    ax2.set_ylabel('Fitness promedio de los individuos')
    ax2.set_title('Evolución del Algoritmo Genético')

    plt.show()

def figuras3(F, points=None, dom=None):
    #if dom is not None:
     #   x = np.linspace(dom[0][0], dom[0][1], 100)
      #  y = np.linspace(dom[1][0], dom[1][1], 100)
    if dom == [[-3,12.1],[4.1,5.8]]:
        x = np.linspace(-4, 14, 100)
        y = np.linspace(10, 35, 100)
    if dom == [[-100.0,100.0],[-100.0,100.0]]:
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
    else:
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    #Evaluar funcion en el mesh
    Z = F([X, Y])

    #Extraer pares x-y
    if points is not None:
        points = np.array(points)
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = F([x_points, y_points])

    #Crear superficie 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)

    #Añadir scattered plots
    if points is not None:
        ax.scatter(x_points, y_points, z_points, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Funcion con pares x-y' if points is not None else 'Funcion a optimizar')

    plt.show()