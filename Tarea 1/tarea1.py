# Tarea 1 - Algoritmos Geneticos para Ingenieros
# Jose Tovar
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
from numpy import sin, cos, pi
import math
 
#Definir rango para entrada
#dom = [[-5.0, 10.0], [-3.0, 9.0], [-1.0, 2.5]] #Funcion 1
dom = [[-8.0, 8.0], [-8.0, 8.0]] #Funcion 0
print("Numero de variables:", len(dom))

# Definir el numero de generaciones
n_iter = 50
# Definir el numero de bits por variable

#Para calcular el numero de bits necesarios dada la precision
pre = 1e-6 #Numero de cifras decimales/ precision

n_bits_array = []

#Para calcular el numero de bits necesarios dada la precision
for i in range(len(dom)):
        numero = (dom[i][1] - dom[i][0])/pre
        n_bits_k = math.ceil(math.log2(numero)) #Redondeo hacia arriba numero de bits necesarios
        n_bits_array.append(n_bits_k)

if len(dom) == 3:
    n_bits_1 = n_bits_array[0]
    n_bits_2 = n_bits_array[1]
    n_bits_3 = n_bits_array[2]
else:
    n_bits_1 = n_bits_array[0]
    n_bits_2 = n_bits_array[1]
    n_bits_3 = 0
print(n_bits_array)
n_bits = sum(n_bits_array)
print("El numero de bits del genotipo es: ", n_bits)

# Tamaño de la poblacion
n_pob = 50
# Variable que controla si se quiere maximizar (1) o minimizar (0) la funcion
tipo_optim = 0
# Variable que controla si se quiere optimizar la F1 (0), F2 (1)
func = 0
# Tasa de crossover segun Holland
r_cross = 0.8
# Tasa de mutacion segun Holland
r_mut = 0.1
# #Parametros de la renormalizacion lineal
# renorm = 0 #Para escoger si se hace la renormalizacion 
# dec_renorm, max_fit = 1, 50


# Funciones objetivo, las cuales se quieren maximizar o minimizar

def F1(x):
     return x[0] + 2*x[1] - 0.3*sin(3*pi*x[0])*0.4*cos(4*pi*x[1]) + 0.4 + 24

def F2(x):
     return (x[0]*x[0]+x[1]*x[1])**0.25*(1+sin(50*(x[0]*x[0]+x[1]*x[1])**0.1)**2) + (x[0]*x[0]+x[2]*x[2])**0.25*(1+sin(50*(x[0]*x[0]+x[2]*x[2])**0.1)**2) + (x[2]*x[2]+x[1]*x[1])**0.25*(1+sin(50*(x[2]*x[2]+x[1]*x[1])**0.1)**2)

# --------------------------Algoritmo Genetico----------------------------------

mejores = []
generaciones = []
indices = []
mejor_par = []
prom = []

def alg_gen(f1, f2, dom, n_bits, n_iter, n_pob, r_cross, r_mut, tipo_optim, func):
    # Se genera poblacion inicial de bit-strings ALEATORIOS
    pob = [randint(0, 2, size=n_bits).tolist() for _ in range(n_pob)]

    print("Primer individuo crudo, sin procesar substring:", pob[0])

    

    # Enumerando generaciones segun el numero de iteraciones de entrada
    for gen in range(n_iter):

        # Se decodifica la poblacion, individuo por individuo
        decoded = [decode(dom, n_bits_1, n_bits_2, n_bits_3, n_bits, p) for p in pob]
        #decoded = [decode(dom, n_bits_array, n_bits, p) for p in pob]

        # Se verifica cual funcion a utilizar
        #Funcion 1
        if func == 0 and tipo_optim == 1:
            fitness = [f1(d) for d in decoded]
        elif func == 0 and tipo_optim == 0:
            fitness = [1/(1+f1(d)) for d in decoded]
        #Funcion 2
        elif func == 1 and tipo_optim == 1:
            fitness = [f2(d) for d in decoded]
        elif func == 1 and tipo_optim == 0:
            fitness = [1/(1+f2(d)) for d in decoded]

        # Se asigna una puntuacion a cada candidato
        # Se busca una solucion mejor entre la poblacion

        # if renorm == 1:
        #     fitness_norm, pob_norm = renorm_lineal(fitness, pob, dec_renorm, max_fit)

        #Buscando el mejor individuo en la generacion actual
        #Necesito el maximo de la generacion actual y el index para ver a cual par pertenece

        best_index, best_eval = np.argmax(fitness), np.amax(fitness)
        best_pair = decoded[best_index]
        promedio = np.mean(fitness)
        mejor_binario = pob[best_index]


        # Se hace la seleccion de los padres recorriendo toda la poblacion
        #Se genera un arreglo de padres seleccionados

        #------------------------Seleccion por Torneo----------------------------------
        #padres_selec = [selection(pob, fitness, tipo_optim)
        #               for _ in range(n_pob)]
        
        #------------------------Seleccion por Ruleta----------------------------------
        padres_selec = ruleta(pob,fitness)
        # if renorm == 1:
        #     padres_selec = ruleta(pob_norm,fitness_norm)
        # else:
        #     padres_selec = ruleta(pob,fitness)


        #padres_selec = uni_estocastica(pob,fitness)

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
        ultimo_mejor = best_pair

        # Se actualiza la poblacion actual SIN ELITISMO
        pob = hijos

    #Mejor individuo de todas las generaciones
    mejor_fitness = np.amax(mejores)
    print("El mejor fitness es: ", mejor_fitness) 
    print("El mejor individuo esta ubicado en la posicion: ", np.argmax(mejores))
    mejor_individuo = mejor_par[np.argmax(mejores)]
    print("El mejor individuo de la ultima generacion esta ubicado en: ", ultimo_mejor)

    return [mejor_individuo, mejor_fitness]

best, puntuacion = alg_gen(F1, F2, dom, n_bits, n_iter,
                            n_pob, r_cross, r_mut, tipo_optim, func)
print('Listo!')
print("El mejor resultado obtenido es el siguiente:", best)

#PRIMERA GRAFICA
fig1 = plt.figure()
plt.plot(generaciones, mejores)
plt.xlabel('Generaciones')
plt.ylabel('Fitness del mejor individuo')
plt.title('Evolución del Algoritmo Genético')

#if func == 0:
#Graficando curva promedio 
fig2 = plt.figure()
plt.plot(generaciones, prom)
plt.xlabel('Generaciones')
plt.ylabel('Fitness promedio de los individuos')
plt.title('Evolución del Algoritmo Genético')
#plt.show()

#SEGUNDA GRAFICA
#Sacando datos del arreglo

if func==0:

    fig3 = plt.figure()
    mejor_par = np.array(mejor_par)
    # Evaluate the F function for each pair
    W = np.zeros(n_iter)
    for i in range(n_iter):
        W[i] = F1(mejor_par[i])
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

#Mostrar las graficas
plt.show()

#Graficando curva promedio 
# fig3 = plt.figure()
# plt.plot(generaciones, prom)
# plt.xlabel('Generaciones')
# plt.ylabel('Fitness promedio de los individuos')
# plt.title('Evolución del Algoritmo Genético')
# # Show the plots
# plt.show()