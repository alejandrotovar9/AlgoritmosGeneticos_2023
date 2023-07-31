import numpy as np

#--------------OPERADORES CON REPRESENTACION BASADA EN ORDEN------------

p1 = [1, 2, 3, 4, 5, 6, 7, 8]
p2 = [8, 6, 4, 2, 7, 5, 3, 1]
ma = [0, 1, 1, 0, 1, 1, 0, 0]


#CRUC UNIFORME
def cruce_uniforme(p1, p2, ma):
    h1 = []
    hijo1 = []
    nuevop2 = []
    for k in range(len(p1)):
        if ma[k] == 1:
            h1.append(p1[k])
        else:
            h1.append('a')
        if p2[k] not in h1:
            nuevop2.append(p2[k])

    for valor in h1:
        if valor == 'a':
            hijo1.append(nuevop2[0])
            nuevop2.pop(0)
        else:
            hijo1.append(valor)

    h2 = []
    hijo2 = []
    nuevop1 = []
    for k in range(len(p1)):
        if ma[k] == 0:
            h2.append(p2[k])
        else:
            h2.append('a')
        if p1[k] not in h2:
            nuevop1.append(p1[k])

    for valor in h2:
        if valor == 'a':
            hijo2.append(nuevop1[0])
            nuevop1.pop(0)
        else:
            hijo2.append(valor)

    return hijo1, hijo2

h1, h2 = cruce_uniforme(p1,p2,ma)
print("El hijo 1 es: ", h1)
print("El hijo 2 es: ", h2)

print('   ')

#---------------------CRUCE OX1-----------------------------

# po1 = [1, 2, 3, 4, 5, 6, 7, 8]
# po2 = [2, 4, 6, 8, 7, 5, 3, 1]
# inicio = 3
# fin = 5

# def OX1(p1,p2):
#     h1 = [0 for i in range(len(p1))]
#     h2 = [0 for i in range(len(p2))]

#     #Genero subarreglo
#     sub_p1h1 = p1[inicio-1: fin-1 + 1] 
#     h1[inicio-1: fin-1 + 1] = sub_p1h1 #Guardo en vector hijo
#     sub_p1h2 = p2[inicio-1: fin-1 + 1]
#     h2[inicio-1: fin-1 + 1] = sub_p1h2 #Guardo en vector hijo

#     # Print the sub-p1
#     print(h1, h2)

# OX1(po1,po2)

#MUTACION POR INSERCION
m1 = [1, 2, 3, 4, 5, 6, 7, 8]
copiam1 = m1.copy()

def mut_ins(m1):
    num1 = np.random.randint(1,7)
    num2 = np.random.randint(1,7)
    #Si son iguales sigo generando numeros aleatorios hasta que no lo sean
    while num2 == num1:
        num2 = np.random.randint(1,7)
    # num1 = 4
    # num2 = 7

    print("El numero a insertar es: ", num1)
    print("Se insertara en la posicion del numero: ", num2)

    #Se guarda el indice de ambos valores
    ind_eliminado = m1.index(num1)
    ind_nuevo = m1.index(num2)

    #Elimina el valor de su posicion actual
    m1.remove(num1)

    #Inserta valor en nuevo indice
    m1.insert(ind_nuevo, num1)

    return m1

mutado = mut_ins(copiam1)
print("El individuo antes de la mutacion por insercion: ", m1)
print("El nuevo individuo despues de la mutacion por insercion: ", mutado)

print('   ')

#MUTACION INVERSION SIMPLE
m2 = [1, 2, 3, 4, 5, 6, 7, 8]
copiam2 = m2.copy()

def mut_inv_simple(m1):
    #inicio = 3
    #fin = 5
    inicio = np.random.randint(0,7)
    fin = np.random.randint(1,7)
    while inicio == fin or inicio > fin:
        fin = np.random.randint(1,7)

    #Genero substring
    sub_m1 = m1[inicio-1: fin-1 + 1]
    print("Substring aleatorio a invertir", sub_m1)
    #Invierto el substring
    sub_inv = list(reversed(sub_m1))
    #Guardo en vector original
    m1[inicio-1: fin-1 + 1] = sub_inv 

    return m1

mutado2 = mut_inv_simple(m2)
print("El individuo antes de la mutacion por inversion simple: ", copiam2)
print("El nuevo individuo despues de la mutacion por inversion simple: ", mutado2)


