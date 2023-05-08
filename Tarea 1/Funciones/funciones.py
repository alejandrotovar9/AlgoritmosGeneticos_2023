# Importando librerias necesarias
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

def decode(dom, n_bits_1, n_bits_2, n_bits, bitstring):
    decodificado = list()  # Se inicializa una lista
    mayor_valor = 2**n_bits
    # Itero para cada variable
    for i in range(len(dom)):
        # Se extrae el substring

        # Se separa el substring en dos cadenas que representan cada variable
        start, end = i * n_bits_1, (i * n_bits_2)+n_bits_1
        substring = bitstring[start:end]

        # Si el substring es mas corto que el numero de bits total, se agregan 0s al final
        while len(substring) < n_bits:
            # Cambiar a los mas significartivos --------------------
            substring.append(0)

        # print("El substring generado en la iteracion es:", substring) BORRAR------------------

        # Se convierte el substring a un string de chars
        # Esto nos da el valor en binario
        chars = ''.join([str(s) for s in substring])
        # Se convierte el string a integer
        integer = int(chars, 2)

        # Se escala el integer a un valor dentro del rango deseado scale integer to desired range
        valor = dom[i][0] + (integer/mayor_valor) * (dom[i][1] - dom[i][0])

        # Guardo en la lista inicial
        decodificado.append(valor)
    return decodificado

# Torneo de Seleccion (se lleva a cabo luego de calcular que tan buena es cada funcion)


def selection(pob, fitness, tipo_optim, k=3):  # k representa el numero de padres_selec
    # Primera seleccion aleatoria
    selection_ix = randint(len(pob))
    for ix in randint(0, len(pob), k-1):  # Escojo un indice en particular de la poblacion
        # Chequear si hay alguno mejor (hacer el torneo)
        # Esto dependera de si se esta maximizando o minimizando
        if tipo_optim == 1:
            if fitness[ix] > fitness[selection_ix]:
                selection_ix = ix
        else:
            if fitness[ix] < fitness[selection_ix]:
                selection_ix = ix

    return pob[selection_ix]

# Crossover simple entre dos padres_selec para crear 2 hijos


def crossover(p1, p2, r_cross):
    # Los hijos son copias de los padres_selec por default
    c1, c2 = p1.copy(), p2.copy()
    # Se evalua la probabilidad de cruce
    if rand() < r_cross:
        # Se selecciona punto de quiebre o punto de cruce distinto de la ultima posicion
        pt = randint(1, len(p1)-2)
        # Se lleva a cabo el cruce manejando la informacion de los vectores, cabeza con cola y viceversa
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# ---------------------------Operador de mutacion


def mutacion(bitstring, r_mut):
    for i in range(len(bitstring)):
        # Se evalua la probabilidad de mutacion
        if rand() < r_mut:
            # Se cambia el valor de un bit en la posicion escogida en caso de que se cumpla condicion de mutacion
            # Si era 0 cambia a 1 y 1 cambia a 0, es el negado
            bitstring[i] = 1 - bitstring[i]
