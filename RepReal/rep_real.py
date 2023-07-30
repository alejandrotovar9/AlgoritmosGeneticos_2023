from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi
import math
import random

# PRECISION
# Para calcular el numero de bits necesarios dada la precision
pre = 1e-6  # Numero de cifras decimales/ precision
n_bits_array = []

dom = [-1.0, 2.0]  # aj y bj


def F1(x):
    return x*sin(10*pi*x) + 1.0


# Para calcular el numero de bits necesarios dada la precision
for i in range(len(dom)):
        numero = (dom[1] - dom[0])/pre
        # Redondeo hacia arriba numero de bits necesarios
        n_bits_k = math.ceil(math.log2(numero))

print(n_bits_k)
n_pob = 2

# Se genera poblacion inicial de bit-strings ALEATORIOS
pob = [randint(0, 2, size=n_bits_k).tolist() for _ in range(n_pob)]

# --------------------OPERADORES DE CRUCE REAL--------------------------------
p1 = [1.07, 2.1, 4.08, 0.7]
p2 = [2.01, 1.8, 3.82, 0.9]
hijo_unico = []

print(range(len(p1)))

# FLAT


def flat(p1, p2):
      for k in range(len(p1)):
            h = random.uniform(p1[k], p2[k])
            truncado = f"{h:.3f}"
            ht = float(truncado)
            hijo_unico.append(ht)
      return hijo_unico


hijo_flat = flat(p1, p2)
print("Hijo producto del flat:", hijo_flat)

# Cruce Aritmetico
K = 0.1  # Factor Lambda


def cruce_arit(p1, p2):
    # Se inicializan vectores vacios
    hijo1 = []
    hijo2 = []

    for k in range(len(p1)):
        h1 = K*p1[k] + (1-K)*p2[k]
        h2 = K*p2[k] + (1-K)*p1[k]
        truncado1 = f"{h1:.3f}"
        h1t = float(truncado1)
        truncado2 = f"{h2:.3f}"
        h2t = float(truncado2)
        hijo1.append(h1t)
        hijo2.append(h2t)

    return hijo1, hijo2


hijo_arit1, hijo_arit2 = cruce_arit(p1, p2)
print("Hijo 1 por cruce aritmetico: ", hijo_arit1, "  |   Hijo 2 por cruce aritmetico: ", hijo_arit2)

# Cruce BLX-alfa
alfa = 0.1


def blx_alfa(p1, p2):

    hijo = []
    for k in range(len(p1)):
          Cmin = min(p1[k], p2[k])
          Cmax = max(p1[k], p2[k])
          #Calculo de intervalos
          a = Cmin - (Cmax - Cmin)*alfa
          b = Cmax + (Cmax - Cmin)*alfa
          #print("a: ", a, "|  b:", b)
          h = random.uniform(a, b)
          truncado = f"{h:.3f}"
          ht = float(truncado)
          hijo.append(ht)

    return hijo

hijo_blx = blx_alfa(p1, p2)
print("Hijo producto del BLX-alfa:", hijo_blx)

#Cruce Lineal
def cruce_lineal(p1, p2):
    hijo1 = []
    hijo2 = []
    hijo3 = []

    for k in range(len(p1)):
        h1i = 0.5*p1[k] + 0.5*p2[k]
        h2i = 1.5*p1[k] - 0.5*p2[k]
        h3i = -0.5*p1[k] + 1.5*p2[k]

        truncado1 = f"{h1i:.3f}"
        h1t = float(truncado1)
        truncado2 = f"{h2i:.3f}"
        h2t = float(truncado2)
        truncado3 = f"{h3i:.3f}"
        h3t = float(truncado3)

        hijo1.append(h1t)
        hijo2.append(h2t)
        hijo3.append(h3t)

    return hijo1, hijo2, hijo3

hijo_lineal1, hijo_lineal2, hijo_lineal3 = cruce_lineal(p1, p2)
print("Hijo 1 lineal: ", hijo_lineal1, "  |   Hijo 2 lineal: ", hijo_lineal2, "  |   Hijo 3 lineal: ", hijo_lineal3)

#Cruce Linea Extendida
def cruce_lin_ext(p1, p2):
    hijo = []
    alf = random.uniform(-0.25, 1.25)
    for k in range(len(p1)):
          hi = p1[k] + alf*(p2[k] - p1[k])

          truncado1 = f"{hi:.3f}"
          hit = float(truncado1)
          hijo.append(hit)
    return hijo

hijo_lin_ext = cruce_lin_ext(p1, p2)
print("Hijo producto de cruce de linea extendida:", hijo_lin_ext)

#Cruce Heuristico de Wright
def cruce_wright(p1,p2):
    hijo = []
    r = np.random.random()
    for k in range(len(p1)):
         hi = p1[k] + r*(p1[k] - p2[k])
         truncado1 = f"{hi:.3f}"
         hit = float(truncado1)
         hijo.append(hit)

    return hijo

hijo_wright = cruce_wright(p1, p2)
print("Hijo producto de cruce heuristico de Wright:", hijo_wright)