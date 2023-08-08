# Tarea 2 - Algoritmos Geneticos para Ingenieros
# Jose Tovar
# Profa. Tamara Perez
# E.I.E - UCV

# Importando librerias necesarias

from Funciones.funciones2 import *  # Importando archivo que contiene las funciones
from Graficas.graficas2 import *

from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi
import math
import csv
 
#Definir rango para entrada
dom = [[-10.0,10.0],[-10.0,10.0]] #F1 Minimizar
#dom = [[-100.0,100.0],[-100.0,100.0]] #F2 Maximizar
#dom = [[-3,12.1],[4.1,5.8]] #F3 Maximizar

# Variable que controla si se quiere maximizar (1) o minimizar (0) la funcion
tipo_optim = 0
# Variable que controla si se quiere optimizar la F1 (1), F2 (2) o F3(3)
func = 1
# Tamaño de la poblacion
n_pob = 50
# Definir el numero de generaciones
n_iter = 100
# Tasa de crossover segun Holland
r_cross = 0.8
# Tasa de mutacion segun Holland
r_mut = 0.02
# Elitismo (1) o sin elitismo (0)
elit = 1
#Parametros de la renormalizacion lineal
renorm = 1 #Para escoger si se hace la renormalizacion 
dec_renorm, max_fit = 1, 100
#GAP GENERACIONAL. Activado (1), Desactivado (0)
gap_gen = 1
#Porcentaje de la poblacion que se cambia
p_gap = 2
#Sin duplicado (1), permite la existencia de duplicados (0)
dupli = 1

#Para calcular el numero de bits necesarios dada la precision
pre = 1e-4 #Numero de cifras decimales/ precision

print("Numero de variables:", len(dom))

n_bits_array = []

#Para calcular el numero de bits necesarios dada la precision\
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
n_bits = sum(n_bits_array)

print("El numero de bits del genotipo es: ", n_bits)
print("Se esta aplicando el AG a la Funcion", func)
print("La funcion se esta", "maximizando." if tipo_optim == 1 else "minimizando." if 
      tipo_optim == 0 else "Introduzca una operacion correcta")
if gap_gen == 1:
    print("Se cambiaran esta cantidad de individuos por el gap generacional: ", int((n_pob)*(p_gap/100)))


# Funciones objetivo, las cuales se quieren maximizar o minimizar

def F1(x):
    return x[0]**2 + 2*(x[1]**2) - 0.3*cos(3*pi*x[0]) - 0.4*cos(4*pi*x[1]) + 0.7

def F2(x):
    return 0.5 - (sin(np.sqrt(x[0]**2+x[1]**2))**2 - 0.5)/(1 + 0.001*(x[0]**2 + x[1]**2))**2

def F3(x):
    return 21.5 + x[0]*sin(4*pi*x[0]) + x[1]*sin(20*pi*x[1])

# --------------------------Algoritmo Genetico----------------------------------

mejores = []
generaciones = []
indices = []
mejor_par = []
prom = []
pob_final =[]

def alg_gen(f1, f2, f3, dom, n_bits, n_iter, n_pob, r_cross, r_mut, tipo_optim, func, elit):
    # Se genera poblacion inicial de bit-strings ALEATORIOS
    pob = [randint(0, 2, size=n_bits).tolist() for _ in range(n_pob)]
    
    # Enumerando generaciones segun el numero de iteraciones de entrada
    for gen in range(n_iter):

        # Se decodifica la poblacion, individuo por individuo
        decoded = [decode(dom, n_bits_1, n_bits_2, n_bits_3, n_bits, p) for p in pob]
        #decoded = [decode(dom, n_bits_array, n_bits, p) for p in pob]

        print(decoded)
        # Se verifica cual funcion a utilizar                

        #Se calcula el fitness para cada individuo (constituido por pares o  de numeros)
        #Funcion 1
        if func == 1 and tipo_optim == 1:
            fitness = [f1(d) for d in decoded]
        elif func == 1 and tipo_optim == 0:
            fitness = [1/(1+f1(d)) for d in decoded]
        #Funcion 2
        elif func == 2 and tipo_optim == 1:
            fitness = [f2(d) for d in decoded]
        elif func == 2 and tipo_optim == 0:
            fitness = [1/(1+f2(d)) for d in decoded]
        #Funcion 3
        elif func == 3 and tipo_optim == 1:
            fitness = [f3(d) for d in decoded]
        elif func == 3 and tipo_optim == 0:
            fitness = [1/(1+f3(d)) for d in decoded]

        fitness = np.around(fitness,15)
        # Se asigna una puntuacion a cada candidato
        # Se busca una solucion mejor entre la poblacion

        #Arreglo la poblacion de mejor a peor para varios usos
        mejor_a_peor_arr = orden_poblacion(fitness, pob)

        if renorm == 1:
            #Se genera vector de fitness y poblacion renormalizado
            fitness_norm, pob_norm, indices_norm = renorm_lineal(fitness, pob, dec_renorm, max_fit)
        
        #Se guardan mejores valores de población actual
        best_index, best_eval = np.argmax(fitness), np.amax(fitness)
        best_pair = decoded[best_index]
        promedio = np.average(fitness)
        mejor_binario = pob[best_index]

        #print("El mejor individuo es: ", mejor_binario)
        #print("El peor individuo es: ", peores_indiv[-1])

        # Se hace la seleccion de los padres recorriendo toda la poblacion
        #------------------------Seleccion por Torneo----------------------------------
        #padres_selec = [selection(pob, fitness, tipo_optim)
        #               for _ in range(n_pob)]
        
        #------------------------Seleccion----------------------------------
        if renorm == 1:
            #padres_selec = ruleta(pob_norm,fitness_norm)
            padres_selec = uni_estocastica(pob_norm,fitness_norm)
            #padres_selec = [selection(pob_norm, fitness_norm, tipo_optim) for _ in range(n_pob)]
        else:
            #padres_selec = ruleta(pob, fitness)
            padres_selec = uni_estocastica(pob,fitness)
            #padres_selec = [selection(pob, fitness, tipo_optim)
        #               for _ in range(n_pob)]
    

        # Se crea la siguiente generacion
        hijos = list()

        #----------------------CRUCE Y MUTACION-------------------------------------------
        for i in range(0, n_pob, 2):  # Pasos de dos
            # Se arreglan los padres seleccionados en pares
            p1, p2 = padres_selec[i], padres_selec[i+1]
            # Cruce y mutacion
            for c in crossover(p1, p2, r_cross):
                # Se ejecuta la mutacion
                mutacion(c, r_mut)
                # Se guarda para la siguiente generacion
                hijos.append(c)

        #Se guardan valores necesarios para las graficas
        generaciones.append(gen)
        indices.append(best_index)
        mejores.append(best_eval)
        mejor_par.append(best_pair)
        prom.append(promedio)
        ultimo_mejor = best_pair

        #print("Peores individuos:", mejor_a_peor_arr[-10:])
        #-----------------------------------------Gap Generacional----------------------------------------
        #Se sustituyen solo un porcentaje de la poblacion generada
        if gap_gen==1:
            for k in range(len(pob) - 1, len(pob) - int((len(pob)+1)*(p_gap/100)) - 1, -1): #Recorre desde el ultimo individuo

                #Genero numero aleatoria que sera la posicion en la cual escogere al hijo nuevo
                num_rand = randint(0, len(pob)- 1)
                #Teniendo el vector de indices, lo recorro desde el peor y sustituyo ese indice en la pob original

                #-----------------------------SIN DUPLICADOS----------------------------------------
                if dupli == 1:
                    while hijos[num_rand] in pob:
                        #Si el hijo a escoger ya se encuentra en la poblacion, se busca un nuevo hijo entre los disponibles
                        num_rand = randint(0, len(pob)- 1)
                #Sustituyo a los peores individuos de la original para nuevos hijos generados 
                pob[mejor_a_peor_arr[k]] = hijos[num_rand] #Creacion de la nueva poblacion
        else:
            # Se actualiza la poblacion actual
            pob = hijos

        #Duplicados en caso de que este desactivada la sustitucion parcial.
        if dupli == 1 and gap_gen == 0:
            for k in range(len(pob)-1):
                if hijos[k] in pob:
                    while hijos[k] in pob:
                        #Si el hijo a escoger ya se encuentra en la poblacion, se busca un nuevo hijo entre los disponibles
                        num_rand = randint(0, len(pob)- 1)
                    pob[k] = hijos[num_rand]
                #Si el valor del hijo es distinto, actualizo poblacion
                else:
                    # Se actualiza la poblacion actual
                    pob[k] = hijos[k]

        #El duplicado lo imprimo despues de mutacion y cruce
            
        #Actualizacion por elitismo, mantengo al mejor individuo de la poblacion anterior
        if elit == 1:
            num_ale = randint(0, len(pob)- 1)
            pob[num_ale] = mejor_binario #sustituyo un individuo de la nueva por el mejor de la generacion anterior
        else:
            continue

    #Mejor individuo de todas las generaciones
    mejor_fitness = np.amax(mejores)
    print("El mejor fitness es: ", mejor_fitness) 
    print("El mejor individuo esta ubicado en la posicion: ", np.argmax(mejores))
    mejor_individuo = mejor_par[np.argmax(mejores)]
    print("El mejor individuo esta en: ", mejor_individuo)

    return [mejor_individuo, mejor_fitness, mejores, prom, generaciones]

#----------------FIN DEL AG---------------------------

#Escoger la funcion a graficar

if func == 1:
    plot_func = F1
elif func == 2:
    plot_func = F2
elif func == 3:
    plot_func = F3

#-----------------CORRIDAS---------------

corridas = 1
mejores_ind_corridas = []
fitness_corridas = []

vector_max_fitness_corridas = []
vector_de_prom_corridas = []

vector_promedio_max_fitness_corridas = []
vector_promedio_prom_corridas = []

for k in range(corridas):

    print("Corrida:", k+1)

    best, puntuacion, vector_mejores_fitness, vector_prom, genera = alg_gen(F1, F2, F3, dom, n_bits, n_iter,
                                n_pob, r_cross, r_mut, tipo_optim, func, elit)
    #BORRO LOS MEJORES DE LA ANTERIOR
    mejores = []
    prom = []
    generaciones = []
    #Mejor individuo encontrado en todas las corridas y todas las generaciones
    np.array(vector_mejores_fitness)
    mejores_ind_corridas.append(best)
    fitness_corridas.append(puntuacion)
    vector_max_fitness_corridas.append(vector_mejores_fitness)
    vector_de_prom_corridas.append(vector_prom)

#Se generan vectores promedio para graficas dado el numero de corridas
vector_promedio_max_fitness_corridas = promedio_corridas(vector_max_fitness_corridas, corridas)
vector_promedio_prom_corridas = promedio_corridas(vector_de_prom_corridas, corridas)

mejor_fitness_total = np.amax(fitness_corridas)
mejor_individuo_total = mejores_ind_corridas[np.argmax(fitness_corridas)]

print("El mejor individuo de todas las corridas tiene un fitness de: ", mejor_fitness_total)
print("El mejor individuo de todas las corridas esta ubicado en la posicion: ", mejor_individuo_total)


print("Las longitudes son:")
print("V1", len(vector_promedio_max_fitness_corridas))
print("V2", len(vector_promedio_prom_corridas))


figuras2(genera, vector_promedio_max_fitness_corridas, vector_promedio_prom_corridas)

print('Listo!')

#Funciones para graficar
# figuras1(mejor_par, n_iter, plot_func)
#figuras2(generaciones, mejores, prom)
# figuras3(plot_func, mejor_par, dom)

