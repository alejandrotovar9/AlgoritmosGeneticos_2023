# Proyecto Final - Algoritmos Geneticos para Ingenieros
# Jose Tovar 07/08/2023
# Profa. Tamara Perez
# E.I.E - UCV

# Importando librerias necesarias

from Funciones.funciones_pid import *  # Importando archivo que contiene las funciones

from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi
import math
import time

#Definir rango para entrada
dom = [[1.0, 100.0], [1.0, 100.0], [1.0, 50.0]] #Funcion 1
print("Numero de variables:", len(dom))

# Definir el numero de generaciones
n_iter = 20
# Definir el numero de bits por variable

#Para calcular el numero de bits necesarios dada la precision
pre = 1e-3 #Numero de cifras decimales/ precision

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
# Tasa de crossover segun Holland
r_cross = 0.8
# Tasa de mutacion segun Holland
r_mut = 0.1

#MODIFICACIONES OPCIONALES AL AG
#Renormalizacion. ON(1) OFF(0) 
renorm = 1
dec_renorm, max_fit = 1, 100
#Elitismo
elit = 1
#Gap generacional
gap_gen = 1
p_gap = 10
dupli = 1

#Contador para las graficas iniciales
count = 0

#Funcion de Transferencia del sistema
num = 0.21
den = [1,0,0]

#Generador de Error para el Fitness
def F(K, tf_sistema):
    #K = [kp, Ki, Kd]

    #Se ajustan pesos
    w1=5
    w2=5
    w3=3
    w4=3
    w5=5
    w6=5

    #Se ejecuta el sistema para obtener la respuesta
    PID_tf = PID_tf_generator(K)
    #Se ejecuta el step function de la planta controlado por el PID
    Time,yout = PID_Plant_Response(PID_tf,tf_sistema)
    #Funcion de transferencia del sistema a lazo cerrado
    T = ct.feedback(PID_tf*tf_sistema,1)


    #Calculo de los errores
    Yout_np=np.array(yout.copy())
    Error=1-Yout_np
    Absolute_Error=np.abs(Error)
    Square_Error=Error**2
    Overshoot=np.max(Yout_np)-1

    #informacion de la respuesta del sistema
    S = ct.step_info(T)
    #Rise_Time = S["RiseTime"]
    #Settling_time = S["SettlingTime"]

    Settling_time = 0
    #Se integra con dx = 0.01 para los distintos errores
    IAE = np.trapz(Absolute_Error, dx=0.01) #Integral del error absoluto
    ISE = np.trapz(Square_Error, dx=0.01) #Integral del Square Error
    ITAE = np.trapz((Time*Absolute_Error), dx=0.01) #Integral del Error Absoluto por el tiempo
    ITSE = np.trapz((Time*Square_Error), dx=0.01) #Integral del Error absoluto
    #u = senal_de_control(Error, PID_tf, Time)

    #Se calcula funcion de costo de salida
    #Cada uno de los indices penaliza y aumenta la funcion de costo, indicando un peor individuo
    W= w1*IAE + w2*ISE + w3*ITAE + w4*ITSE + w5*Settling_time + w6*Overshoot
    #W = senal_de_control(Error, PID_tf, Time)
    #PENALIZR SENALES DE CONTROL MUY GRANDES

    #print("La funcion F se tardo ", time.time() - start_time_F, "en ejecutarse")

    return W

#Sistema antes del Control PID
ball_beam_sistema = tf_planta_generator(num,den)
tiempo_sin_pid, yout_sin_pid = step_response_sinPID(ball_beam_sistema)
plot_grafica(tiempo_sin_pid, yout_sin_pid, "Respuesta escalón sin control PID", "Tiempo", "Amplitud")
J = ct.step_info(ball_beam_sistema)

# --------------------------Algoritmo Genetico----------------------------------

mejores = []
generaciones = []
indices = []
mejor_par = []
prom = []

def alg_gen(f, dom, n_bits, n_iter, n_pob, r_cross, r_mut, count):
    # Se genera poblacion inicial de bit-strings ALEATORIOS
    pob = [randint(0, 2, size=n_bits).tolist() for _ in range(n_pob)]

    print("Primer individuo crudo, sin procesar substring:", pob[0])

    # Enumerando generaciones segun el numero de iteraciones de entrada
    for gen in range(n_iter):

        # Se decodifica la poblacion, individuo por individuo
        decoded = [decode(dom, n_bits_1, n_bits_2, n_bits_3, n_bits, p) for p in pob]

        #Minimizando la funcion de fitness
        fitness = [1/(1+f(d, ball_beam_sistema)) for d in decoded]

        #Para graficar los primeros individuos generados
        if gen == 0:
            #Grafico los primeros 5 individuos generados en la primera generacion
            for ind in decoded:
                tf_ind = PID_tf_generator(ind)
                #Se ejecuta el step function de la planta controlado por el PID
                tiempo_ind,yout_ind = PID_Plant_Response(tf_ind,ball_beam_sistema)
                plot_grafica(tiempo_ind,yout_ind,"Step Response","Tiempo","Amplitud")
                count = count +1
                if count == 1:
                    break

        # Se asigna una puntuacion a cada candidato

        #Arreglo la poblacion de mejor a peor para varios usos
        mejor_a_peor_arr = orden_poblacion(fitness, pob)

        if renorm == 1:
            #Se genera vector de fitness y poblacion renormalizado
            fitness_norm, pob_norm, indices_norm = renorm_lineal(fitness, pob, dec_renorm, max_fit)
        

        # Se busca una solucion mejor entre la poblacion
        #Buscando el mejor individuo en la generacion actual
        #Necesito el maximo de la generacion actual y el index para ver a cual par pertenece
        best_index, best_eval = np.argmax(fitness), np.amax(fitness)
        best_pair = decoded[best_index]
        promedio = np.mean(fitness)
        mejor_binario = pob[best_index]

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

        # # Se actualiza la poblacion actual SIN ELITISMO
        # pob = hijos

        #Actualizacion por elitismo, mantengo al mejor individuo de la poblacion anterior
        if elit == 1:
            num_ale = randint(0, len(pob)- 1)
            pob[num_ale] = mejor_binario #sustituyo un individuo de la nueva por el mejor de la generacion anterior
        else:
            continue
        
        mejor_individuo = mejor_par[np.argmax(mejores)]

        #Se consigue la funcion de transferencia del sistema para obtener la respuesta
        PID_tf = PID_tf_generator(mejor_individuo)
        #Se ejecuta el step function de la planta controlado por el PID
        Time,yout = PID_Plant_Response(PID_tf,ball_beam_sistema)
        T = ct.feedback(PID_tf*ball_beam_sistema,1)
        S = ct.step_info(T)
        Rise_Time = S["RiseTime"]
        Settling_time = S["SettlingTime"]
        Overshoot = S["Overshoot"]
        #CONDICION DE PARADA Overshoot o tiempo de establecimiento menor a 1 y 1.5 segundos
        if Overshoot <= 0.4 and Settling_time <= 0.5:
            print("  ")
            print("Se consiguio un individuo que cumple con las condiciones en al generacion ", gen)
            break
        else:
            continue


    print("El algoritmo se tardo ", time.time() - start_time, " segundos en ejecutarse")

    #Mejor individuo de todas las generaciones
    mejor_fitness = np.amax(mejores)
    print("El mejor fitness es: ", mejor_fitness) 
    #print("El mejor individuo esta ubicado en la posicion: ", np.argmax(mejores))
    mejor_individuo = mejor_par[np.argmax(mejores)]
    print("El mejor individuo de la ultima generacion esta ubicado en: ", ultimo_mejor)
    print("  ")

    return [mejor_individuo, mejor_fitness]

start_time = time.time()
best, puntuacion = alg_gen(F, dom, n_bits, n_iter,
                            n_pob, r_cross, r_mut, count)
#Graficando respuesta en tiempo del sistema
#Se crea el controlador con la mejor solucion encontrada
PID_tf = PID_tf_generator(best)
#Se ejecuta el step function de la planta controlado por el PID
Time,yout = PID_Plant_Response(PID_tf,ball_beam_sistema)

#Funcion de transferencia del sistema a lazo cerrado
T = ct.feedback(PID_tf*ball_beam_sistema,1)
S = ct.step_info(T)
Rise_Time = S["RiseTime"]
Settling_time = S["SettlingTime"]
Overshoot = S["Overshoot"]
print("  ")
print("Caracteristicas del sistema controlado:")
print("Rise Time:", Rise_Time)
print("Settling Time:", Settling_time)
print("Overshoot:", Overshoot)
print("  ")
#print("El mejor resultado obtenido es el siguiente:", best)
print("Siendo los parametros del PID: Kp = ", best[0], " Ki = ", best[1], "Kd = ", best[2])

plot_grafica(Time,yout,"Step Response","Tiempo","Amplitud")

figuras_pid(generaciones, mejores, prom)