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
import numpy as np

# Funciones objetivo, las cuales se quieren maximizar o minimizar

# AJUSTAR INVERSA PARA CUANDO ESTE MINIMIZNDO


def F1(x):
    return x[0] + 2*x[1] - 0.3*np.sin(3*np.pi*x[0])*0.4*np.cos(4*np.pi*x[1]) + 0.4


def F2(x):
    return (x[0]*x[0]+x[1]*x[1])**0.25*(1+np.sin(50*(x[0]*x[0]+x[1]*x[1])**0.1)**2) + (x[0]*x[0]+x[2]*x[2])**0.25*(1+np.sin(50*(x[0]*x[0]+x[2]*x[2])**0.1)**2) + (x[2]*x[2]+x[1]*x[1])**0.25*(1+np.sin(50*(x[2]*x[2]+x[1]*x[1])**0.1)**2)

# Decodificar bit-strings a numeros


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

# --------------------------Algoritmo Genetico----------------------------------


def alg_gen(f1, f2, dom, n_bits, n_iter, n_pob, r_cross, r_mut, tipo_optim, func):
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

        # Se actualiza la poblacion
        pob = hijos
        # print("Esta es la generacion: ", gen)
    return [best, best_eval]


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
tipo_optim = 0

# Variable que controla si se quiere optimizar la F1 (1) o F2 (0)
func = 1

# Tasa de crossover segun Holland
r_cross = 0.8

# Tasa de mutacion segun Holland
r_mut = 0.1


# Ejecutar el algoritmo genetico

print("El mejor resultado obtenido es el siguiente: ")
best, puntuacion = alg_gen(F1, F2, dom, n_bits, n_iter,
                           n_pob, r_cross, r_mut, tipo_optim, func)

print('Listo!')
decoded = decode(dom, n_bits_1, n_bits_2, n_bits, best)
print('f(%s) = %f' % (decoded, puntuacion))
