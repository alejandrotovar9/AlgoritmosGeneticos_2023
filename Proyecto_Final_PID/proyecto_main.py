#Proyecto Final - Algoritmos Geneticos para Ingenieros
#Escuela de Ingenieria Electrica - UCV
#Jose Alejandro Tovar Briceño

from Funciones.funciones_pid import *  # Importando archivo que contiene las funciones

import control as ct
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy.random import rand
from numpy import sin, cos, pi
import math

#Generador de Error para el Fitness
def F(K):
    # K = [kp, Ki, Kd]
    #Se guarda la funcion de transferencia del sistema
    tf_sistema=tf_planta_generator()
    #Se ajustan pesos
    w1=5
    w2=5
    w3=1
    w4=1
    w5=4
    w6=6

    #Se ejecuta el sistema para obtener la respuesta
    PID_tf=PID_tf_generator(K)
    Time,yout=PID_Plant_Response(PID_tf,tf_sistema)

    #Calculo de los errores
    Yout_np=np.array(yout.copy())
    Error=1-Yout_np
    Absolute_Error=np.abs(Error)
    Square_Error=Error**2
    Overshoot=np.max(Yout_np)-1

    # Overshoot_Index=np.where(Yout_np==(1+Overshoot))[0]
    # Overshoot_location=np.argmax(yout)/10
    #Sliced_Square_Error=Square_Error[:Overshoot_Index[0]]
    #Rise_Time_Index=np.where(Sliced_Square_Error==(np.min(Sliced_Square_Error)))
    #Rise_Time=Time[Rise_Time_Index][0]

    Rise_Time=0
    #Se integra con dx = 0.01 para los distintos errores
    IAE = np.trapz(Absolute_Error, dx=0.01) #Integral del error absoluto
    ISE = np.trapz(Square_Error, dx=0.01) #Integral del Square Error
    ITAE = np.trapz((Time*Absolute_Error), dx=0.01) #Integral del Error Absoluto por el tiempo
    ITSE = np.trapz((Time*Square_Error), dx=0.01) #Integral del Error absolt

    #Se calcula funcion de costo de salida
    W= w1*IAE + w2*ISE + w3*ITAE + w4*ITSE + w5*Rise_Time + w6*Overshoot

    return W

#Creando funcion de transferencia del sistema
s = ct.tf('s')
num = 0.21
den = [1,0,0]
sys = ct.tf(num,den)
print(sys)

Kp = 91
Kd = 999
Ki = 3.7
C = (Kd*s**2 + Kp*s + Ki)/(s)

#Funcion de transferencia del sistema a lazo cerrado
T = ct.feedback(C*sys,1)


#Graficando Step Response
time_vector = np.linspace(0,5,100)
time1,yout1 = ct.step_response(T,time_vector)

Time = time1

 #Se ajustan pesos
w1=5
w2=5
w3=1
w4=1
w5=4
w6=4

#Calculo de los errores
Yout_np=np.array(yout1.copy())
Error=1-Yout_np
Absolute_Error=np.abs(Error)
Square_Error=Error**2
Overshoot=np.max(Yout_np)-1
#informacion de la respuesta del sistema
S = ct.step_info(T)
Rise_Time = S["RiseTime"]
print("Rise Time:", Rise_Time)

#Se integra con dx = 0.01 para los distintos errores
IAE = np.trapz(Absolute_Error, dx=0.01) #Integral del error absoluto
ISE = np.trapz(Square_Error, dx=0.01) #Integral del Square Error
ITAE = np.trapz((Time*Absolute_Error), dx=0.01) #Integral del Error Absoluto por el tiempo
ITSE = np.trapz((Time*Square_Error), dx=0.01) #Integral del Error absolt

# ISE integrates the square of the error over time. 
# ISE will penalise large errors more than smaller ones 
# (since the square of a large error will be much bigger). 
# Control systems specified to minimise ISE will tend to eliminate large errors quickly,
#  but will tolerate small errors persisting for a long period of time. O
# ften this leads to fast responses, but with considerable, low amplitude, oscillation.

# IAE integrates the absolute error over time. 
# It doesn't add weight to any of the errors in a systems response. 
# It tends to produce slower response than ISE optimal systems, 
# but usually with less sustained oscillation.

# ITAE integrates the absolute error multiplied by the time over time. 
# What this does is to weight errors which exist after a long time much 
# more heavily than those at the start of the response. 
# ITAE tuning produces systems which settle much more quickly than the other two tuning methods. 
# The downside of this is that ITAE tuning also produces systems 
# with sluggish initial response (necessary to avoid sustained oscillation)

print(Error)
print("IAE ", IAE)
print("ISE ", ISE)
print("ITAE ", ITAE)
print("ITSE ", ITSE)

#Se calcula funcion de costo de salida
W= w1*IAE + w2*ISE + w3*ITAE + w4*ITSE + w5*Rise_Time + w6*Overshoot
print("W ", W)


plot_grafica(time1,yout1,"Step Response","Tiempo","Amplitud")
