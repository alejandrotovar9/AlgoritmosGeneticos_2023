# Tarea 1 - Algoritmos Geneticos para Ingenieros
# Jose Tovar - jatovar02@gmail.com
# Profa. Tamara Perez
# E.I.E - UCV

# PARAMETROS:
# a) la escogencia de la función a optimizar
# b) la selección de la tarea a realizar (maximizar o minimizar)
# c) el tamaño de la población
# d) la cantidad de generaciones
# e) probabilidades de cruce y mutación
# f) número de bits del cromosoma (incluyendo número de bits asignados a cada variable, permitiendo que sean diferentes)
# g) número de corridas.

# Importando librerias necesarias

from Funciones.funciones import *  # Importando archivo que contiene las funciones

from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

# Definir rango para entrada
dom = [[-8.0, 8.0], [-8.0, 8.0]]
print("Numero de variables:", len(dom))
# Definir el numero de generaciones
n_iter = 100
# Definir el numero de bits por variable
n_bits_1 = 10
n_bits_2 = 10
n_bits = n_bits_1 + n_bits_2
# Tamaño de la poblacion
n_pob = 50
# Variable que controla si se quiere maximizar (1) o minimizar (0) la funcion
tipo_optim = 1
# Variable que controla si se quiere optimizar la F1 (1) o F2 (0)
func = 1
# Tasa de crossover segun Holland
r_cross = 0.8
# Tasa de mutacion segun Holland
r_mut = 0.1
# Elitismo (1) o sin elitismo (0)
elit = 0

# Funciones objetivo, las cuales se quieren maximizar o minimizar
# AJUSTAR INVERSA PARA CUANDO ESTE MINIMIZNDO


def F1(x):
    return x[0] + 2*x[1] - 0.3*np.sin(3*np.pi*x[0])*0.4*np.cos(4*np.pi*x[1]) + 0.4


def F2(x):
    return (x[0]*x[0]+x[1]*x[1])**0.25*(1+np.sin(50*(x[0]*x[0]+x[1]*x[1])**0.1)**2) + (x[0]*x[0]+x[2]*x[2])**0.25*(1+np.sin(50*(x[0]*x[0]+x[2]*x[2])**0.1)**2) + (x[2]*x[2]+x[1]*x[1])**0.25*(1+np.sin(50*(x[2]*x[2]+x[1]*x[1])**0.1)**2)

# --------------------------Algoritmo Genetico----------------------------------

mejores = []
generaciones = []
mejor_individuo_x = []
mejor_individuo_y = []
indices = []
mejor_par = []
prom = []

def alg_gen(f1, f2, dom, n_bits, n_iter, n_pob, r_cross, r_mut, tipo_optim, func, elit):
    # Se genera poblacion inicial de bit-strings ALEATORIOS
    pob = [randint(0, 2, size=n_bits).tolist() for _ in range(n_pob)]

    print("Primer individuo crudo, sin procesar substring:", pob[0])
    print("Salida del decodificador para el primer individuo: ",
          decode(dom, n_bits_1, n_bits_2, n_bits, pob[0]))

    # Se guarda la mejor solucion inicial dependiendo de la funcion a estudiar
    if func == 1:
        best, best_eval = 0, f1(
            decode(dom, n_bits_1, n_bits_2, n_bits, pob[0]))
    else:
        fbest, best_eval = 0, f2(
            decode(dom, n_bits_1, n_bits_2, n_bits, pob[0]))

    # Enumerando generaciones segun el numero de iteraciones de entrada
    for gen in range(n_iter):

        # Se decodifica la poblacion, individuo por individuo
        decoded = [decode(dom, n_bits_1, n_bits_2, n_bits, p) for p in pob]

        # Se verifica cual funcion a utilizar
        if func == 1:
            fitness = [f1(d) for d in decoded]
        else:
            fitness = [f2(d) for d in decoded]

        # Se asigna una puntuacion a cada candidato
        # Se busca una solucion mejor entre la poblacion

        # Elitismo
        if elit == 1:
            if tipo_optim == 1:
                for i in range(n_pob):
                    if fitness[i] > best_eval:  # Esta linea define si busco el maximo o el minimo
                        best, best_eval = pob[i], fitness[i]
                        print(">%d, nuevo mejor f(%s) = %f" %
                              (gen, decoded[i], fitness[i]))
            else:
                for i in range(n_pob):
                    if fitness[i] < best_eval:  # Esta linea define si busco el maximo o el minimo
                        best, best_eval = pob[i], fitness[i]
                        print(">%d, nuevo mejor f(%s) = %f" %
                              (gen, decoded[i], fitness[i]))
        #Sin elitismo, busco el mejor en cada generacion y actualizo
        else:
            # Necesito el maximo de la generacion actual y el index para ver a cual par pertenece
            best_index, best_eval = np.argmax(fitness), np.amax(fitness)
            best_pair = decoded[best_index]
            promedio = np.mean(fitness)

            print("Mejor indice:", best_index)
            print("Mejor fitness: ", best_eval)
            print("Mejor par: ", best_pair)

        # Se hace la seleccion de los padres
        padres_selec = [selection(pob, fitness, tipo_optim)
                        for _ in range(n_pob)]

        # Se crea la siguiente generacion
        hijos = list()

        for i in range(0, n_pob, 2):  # Pasos de dos
            # Se arreglan los padres seleccionados en pares
            p1, p2 = padres_selec[i], padres_selec[i+1]
            # Cruce y mutacion
            for c in crossover(p1, p2, r_cross):
                # Se ejecuta la mutacion
                mutacion(c, r_mut)
                # Se guarda para la siguiente generacion
                hijos.append(c)

        generaciones.append(gen)
        indices.append(best_index)
        mejores.append(best_eval)
        mejor_par.append(best_pair)
        prom.append(promedio)
        best = best_pair

        # Se actualiza la poblacion
        pob = hijos
        
    return [best, best_eval]


best, puntuacion = alg_gen(F1, F2, dom, n_bits, n_iter,
                           n_pob, r_cross, r_mut, tipo_optim, func, elit)

print('Listo!')

print("El mejor resultado obtenido es el siguiente: ", best)

#Los mejores individuos estan ubicados en las posiciones asignadas por best_index en el arreglo de la poblacion

#PRIMERA GRAFICA
fig1 = plt.figure()
plt.plot(generaciones, mejores)
plt.xlabel('Generaciones')
plt.ylabel('Fitness del mejor individuo')
plt.title('Evolución del Algoritmo Genético')


#SEGUNDA GRAFICA
#Sacando datos del arreglo

fig2 = plt.figure()
mejor_par = np.array(mejor_par)

# Evaluate the F function for each pair
W = np.zeros(100)
for i in range(100):
    W[i] = F1(mejor_par[i])

# Create a scatter plot of the x-y value pairs with the W values as color
plt.scatter(mejor_par[:, 0], mejor_par[:, 1], c=W, s=20)

#identificando los puntos
# Add text labels to the plot
for i in range(100):
    plt.text(mejor_par[i, 0], mejor_par[i, 1], i+1, ha='center', va='center')

# Add colorbar and labels to the plot
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ubicacion de los mejores individuos ubicados en el plano 2D')

fig3 = plt.figure()
plt.plot(generaciones, prom)
plt.xlabel('Generaciones')
plt.ylabel('Fitness promedio de los individuos')
plt.title('Evolución del Algoritmo Genético')

# Show the plots
plt.show()