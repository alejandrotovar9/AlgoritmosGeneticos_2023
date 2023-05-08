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
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

#Constantes
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
r_mut = 0.15
# Elitismo (1) o sin elitismo (0)
elit = 0

def F1(x):
    return x[0] + 2*x[1] - 0.3*np.sin(3*np.pi*x[0])*0.4*np.cos(4*np.pi*x[1]) + 0.4


def F2(x):
    return (x[0]*x[0]+x[1]*x[1])**0.25*(1+np.sin(50*(x[0]*x[0]+x[1]*x[1])**0.1)**2) + (x[0]*x[0]+x[2]*x[2])**0.25*(1+np.sin(50*(x[0]*x[0]+x[2]*x[2])**0.1)**2) + (x[2]*x[2]+x[1]*x[1])**0.25*(1+np.sin(50*(x[2]*x[2]+x[1]*x[1])**0.1)**2)

#Generando poblacion aleatoria

a1 = np.random.rand(50)