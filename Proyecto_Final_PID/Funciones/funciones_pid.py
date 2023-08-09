# Importando librerias necesarias
from numpy.random import randint
import matplotlib.pyplot as plt
from numpy.random import rand
import numpy as np
import control as ct
from control.matlab import lsim

#------------------FUNCIONES DEL PID------------------

def senal_de_control(error, PID_tf, time):
    Q = 1
    R = 1
    control_signal, T, xout = lsim(PID_tf, error, time)
    u = np.array(control_signal.copy())
    J = np.trapz(Q*error**2 + R*(0.001*u**2), dx=0.01)
    return 1/J

#Construye la forma del controlador PID con los parametros del PID como entrada
def PID_tf_generator(K):
    # K = [kp, Ki, Kd]
    # PID_Num=[K[2],K[0],K[1]]
    # PID_Den=[1,0]
    # PID_tf=ct.tf(PID_Num,PID_Den)
    s = ct.tf('s')
    K = K[0] + K[1] + (K[2]*s)/(1+0.001*s)
    PID_tf=ct.tf(K)
    return PID_tf

#Ejecuta la respuesta escalon del sistema y nos devuelve salida y vector de tiempo
def PID_Plant_Response(PID_TF,TF_gr):
    feedBack = ct.feedback(PID_TF*TF_gr,1)
    #Time=list(np.arange(0,40,0.1)) #400 puntos
    time = np.linspace(0,5,100)
    time, yout = ct.step_response(sys=feedBack,T=time,X0=0)
    return time,yout

def step_response_sinPID(sys):
    time = np.linspace(0,5,100)
    time, yout = ct.step_response(sys,T=time,X0=0)
    return time,yout

#Crea la funcion de transferencia de la planta o Ball and Beam
def tf_planta_generator(num,den):
    s = ct.tf('s')
    sys = ct.tf(num,den)
    return sys

def plot_grafica(tiempo, yout, titulo, titulox, tituloy):
    plt.figure(figsize=(8,6))
    plt.plot(tiempo, yout, color = 'blue', linewidth=1)
    plt.title(titulo)
    plt.xlabel(titulox)
    plt.ylabel(tituloy)
    plt.tick_params(axis='both',which='major',labelsize=14)
    plt.grid(visible=True)
    plt.autoscale(enable=True, axis='x') 
    plt.show()

def decode(dom, n_bits_1, n_bits_2, n_bits_3, n_bits, bitstring):

    decodificado = list()  # Se inicializa una lista
    mayor_valor = 2**n_bits
    # Itero para cada variable
    for i in range(len(dom)):
        # Se extrae el substring

        # Se separa el substring en dos o tres cadenas que representan cada variable
        if len(dom) == 2:

            start, end = i * n_bits_1, (i * n_bits_2)+n_bits_1
            substring = bitstring[start:end]
        else:

            if i == 0:
                start = 0
                end = n_bits_1
            elif i == 1:
                start = n_bits_1
                end = n_bits_1 + n_bits_2
            elif i == 2:
                start = n_bits_1 + n_bits_2
                end = n_bits_1 + n_bits_2 + n_bits_3
            substring = bitstring[start:end]

        valor_max = 2**len(substring)

        #while len(substring) < n_bits:
         #   substring.append(0)
        
        # Se convierte el substring a un string de chars

        # Esto nos da el valor en binario
        chars = ''.join([str(s) for s in substring])
        # Se convierte el string a integer
        integer = int(chars, 2) 
        #El mayor valor posible vendra dado por el numero de bits del substring antes de llenar de 0
        
        # Se escala el integer a un valor dentro del rango deseado
        valor = dom[i][0] + (integer/(valor_max-1)) * (dom[i][1] - dom[i][0])
        
        rounded_number = round(valor, 5)

        # Guardo en la lista inicial
        decodificado.append(rounded_number)
    return decodificado

#-----------------------PROMEDIO-----------------------
def promedio_corridas(arrays, corridas):
    """
    Get the average of several arrays for each element.

    Args:
        arrays: A list of arrays.

    Returns:
        A list of the averages of the arrays.
    """

    c = np.zeros(len(arrays[0]))
    #recorro todo el vector posicion por posicion
    for i in range(len(arrays[0])):
        average = 0
        for array in arrays:
            average += array[i]
        average = average / len(arrays)
        c[i] = average
    return c

#-----------------------PEORES INDIVIDUOS------------------------------

def orden_poblacion(fitness, pob):
    # Se crea arreglo de index
    indices = list(range(len(fitness)))

    #Se arregla el arreglo de indices de mayor a menor con base en el arreglo1
    indices.sort(key=lambda i: fitness[i], reverse=True)
    
    #Se usa el arreglo de indices arreglados para sortear los 2 originales
    sorted_fitness = [fitness[i] for i in indices]
    sorted_pob = [pob[i] for i in indices]

    #Devuelvo el valor de los indices ordenados, que es el orden de los mejores individuos de mayor a menor

    return indices

#-----------------------RENORMALIZACION LINEAL------------------------------------------

def renorm_lineal(fitness, pob, dec, max_fit):
    # Se crea arreglo de index
    indices = list(range(len(fitness)))

    #Se arregla el arreglo de indices de mayor a menor con base en el arreglo1
    indices.sort(key=lambda i: fitness[i], reverse=True)

    #Se usa el arreglo de indices arreglados para sortear los 2 originales
    sorted_fitness = [fitness[i] for i in indices]
    sorted_pob = [pob[i] for i in indices]

    #Se inicializa arreglo vacio para el fitness renormlaizado
    fit_renorm = []
    num = 0

    for k in range(len(pob)):
        if k == 0:
            num = max_fit
        else:
            num = num - dec
        fit_renorm.append(num)

    return fit_renorm, sorted_pob, indices

#------------------------SELECCION POR TORNEO-------------------------------------------

def selection(pob, fitness, tipo_optim, k=3):  # k representa el numero de padres_selec

    # Primera seleccion aleatoria
    selection_ix = randint(len(pob)) #Indice del individuo seleccionado

    #Un problema del torneo es que selecciono dentro de la misma poblacion, cuando
    #deberia descartar a los que ya ganaron en el torneo pasado

    for ix in randint(0, len(pob), k-1):  # Escojo un indice aleatorio en particular de la poblacion
        #Chequear si hay alguno mejor (hacer el torneo)
        if tipo_optim == 1:
            if fitness[ix] > fitness[selection_ix]:
                selection_ix = ix
        #else:
         #   if fitness[ix] > fitness[selection_ix]:
          #      selection_ix = ix

    return pob[selection_ix]

#----------------------------------------------RULETA-------------------------------------------

def ruleta(pop, fitness):
    total_fitness = sum(fitness)
    prob = [f/total_fitness for f in fitness] #Divido cada valor de fitness sobre el total
    acum_prob = []

    #calculo acumulado del vector de probabilidades hasta el punto k en cada iteracion
    for k in range(len(prob)):
        arr_aux = prob[:k+1]
        acum_prob.append(sum(arr_aux))

    elegidos = []
    
    for i in range(len(pop)):
        r = np.random.rand() #genero num aleatorio
        for j in range(len(acum_prob)):
            if r <= acum_prob[j]:
                elegidos.append(pop[j])
                break
    return elegidos

#-----------------------------UNIVERSAL ESTOCASTICA-------------------------------------

def uni_estocastica(pop, fitness):
    total_fitness = np.sum(fitness)
    prom_fit = total_fitness / len(fitness)
    ei = [fit / prom_fit for fit in fitness] #Arreglo que contiene numero de copias
    ptr = np.random.uniform(0,1) #Genero numero aleatorio
    suma = 0
    i = 1
    elegidos = [] #Arreglo para guardar individuos seleccionados

    while i<=len(pop):
        suma = suma + ei[i-1]
        while suma > ptr:
            elegidos.append(pop[i-1])
            ptr = ptr + 1
        i = i+1
    
    return elegidos

#--------------- Crossover simple entre dos padres_selec para crear 2 hijos-----------

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

def mutacion(bitstring, r_mut):
    for i in range(len(bitstring)):
        # Se evalua la probabilidad de mutacion
        if rand() < r_mut:
            # Se cambia el valor de un bit en la posicion escogida en caso de que se cumpla condicion de mutacion
            # Si era 0 cambia a 1 y 1 cambia a 0, es el negado
            bitstring[i] = 1 - bitstring[i]

#----------------------------------GRAFICAS------------------------------------

def figuras_pid(generaciones, mejores, prom):

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
