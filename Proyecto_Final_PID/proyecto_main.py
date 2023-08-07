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

#Construye la forma del controlador PID con los parametros del PID como entrada
def PID_tf_generator(Kd,Kp,Ki):
    PID_Num=[Kd,Kp,Ki]
    PID_Den=[1,0]
    PID_tf=ct.tf(PID_Num,PID_Den)
    return PID_tf

#Ejecuta la respuesta escalon del sistema y nos devuelve salida y vector de tiempo
def PID_Plant_Response(PID_TF,TF_gr):
    feedBack = ct.feedback(PID_TF*TF_gr,1)
    #Time=list(np.arange(0,40,0.1))
    time = np.linspace(0,5,100)
    time, yout = ct.step_response(sys=feedBack,T=time,X0=0)
    return time,yout

def tf_planta_generator(num,den):
    s = ct.tf('s')
    sys = ct.tf(num,den)
    return sys


#Generador de Error para el Fitness
def BioPlant_Error_Generator(Kd,Kp,Ki):
    #Se guarda la funcion de transferencia del sistema
    tf_sistema=tf_planta_generator()
    #Se ajustan pesos
    w1=5
    w2=5
    w3=1
    w4=1
    w5=4
    w6=0

    #Se ejecuta el sistema para obtener la respuesta
    PID_tf=PID_tf_generator(Kd,Kp,Ki)
    Time,yout=PID_Plant_Response(PID_tf,tf_sistema)

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
    IAE = np.trapz(Absolute_Error, dx=0.01)
    ISE = np.trapz(Square_Error, dx=0.01)
    ITAE = np.trapz((Time*Absolute_Error), dx=0.01)
    ITSE = np.trapz((Time*Square_Error), dx=0.01)

    #Se calcula funcion de costo de salida
    W= w1*IAE + w2*ISE + w3*ITAE + w4*ITSE + w5*Rise_Time + w6*Overshoot

    return W,yout,Time,IAE,ISE,ITAE,ITSE,Rise_Time,Overshoot


#Creando funcion de transferencia del sistema
s = ct.tf('s')
num = 1
den = [1,10,20]
sys = ct.tf(num,den)

Kp = 200
Kd = 50
Ki = 20
C = (Kd*s**2 + Kp*s + Ki)/(s)

#Funcion de transferencia del sistema a lazo cerrado
T = ct.feedback(C*sys,1)


#Graficando Step Response
time_vector = np.linspace(0,5,100)
time1,yout1 = ct.step_response(T,time_vector)
Yout_np=np.array(yout1.copy())
#Error (diferencia entre salida y entrada)
error=1-Yout_np
#Sobrepico (Diferencia maxima entre salida y referenciua)
overshoot=np.max(Yout_np)-1

plot_grafica(time1,yout1,"Step Response","Tiempo","Amplitud")
