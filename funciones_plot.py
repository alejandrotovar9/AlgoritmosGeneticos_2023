import numpy as np
import math

#Definir rango para entrada
dom = [[-5.0, 10.0], [-3.0, 9.0], [-1.0, 2.5]]
#dom = [[-8.0, 8.0], [-8.0, 8.0]]

print("Numero de variables:", len(dom))
# Definir el numero de generaciones
n_iter = 50

# Definir el numero de bits por variable
n_bits_1 = 10
n_bits_2 = 10
n_bits_3 = 10

pre = 1e-4 #Numero de cifras decimales

n_bits_array = []

#Para calcular el numero de bits necesarios dada la precision\
for i in range(len(dom)):
        np = (dom[i][1] - dom[i][0])/pre
        print("Np>",np)
        n_bits_k = math.ceil(math.log2(np)) #Redondeo hacia arriba
        n_bits_array.append(n_bits_k)
'''if len(dom) == 3:
    for i in range(len(dom)):
        np = (dom[i][1] - dom[i][0])/pre
        n_bits_k = math.ceil(math.log2(np)) #Redondeo hacia arriba
        n_bits_array.append(n_bits_k)
else:
    for i in range(len(dom)):
        np = (dom[i][1] - dom[i][0])/pre
        n_bits_k = math.ceil(math.log2(np)) #Redondeo hacia arriba
        n_bits_array.append(n_bits_k)
'''

print(n_bits_array[2])
print(n_bits_array)

n_bits_array = np.array(n_bits_array)

#Mejor que se escoja la precision y se calcule el numero de bits necesarios a partir del numero de bits. OJO


# Define the range of x values
#x_min, x_max = -8, 8
#y_min, y_max = -8, 8
#n_points = 100
#x_values = np.linspace(x_min, x_max, n_points)
#y_values = np.linspace(y_min, y_max, n_points)

# Create a meshgrid of x and y values
#X, Y = np.meshgrid(x_values, y_values)

# Evaluate the function at each point in the meshgrid
#Z = F1([X, Y])

# Create a 2D contour plot of the function
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].contourf(X, Y, Z, levels=30, cmap='cool')
#ax[0].set_xlabel('x1')
#ax[0].set_ylabel('x2')
#ax[0].set_title('Contour plot of F1(x)')

# Create a 3D surface plot of the function
#ax[1] = fig.add_subplot(122, projection='3d')
#ax[1].plot_surface(X, Y, Z, cmap='cool')
#ax[1].set_xlabel('x1')
#ax[1].set_ylabel('x2')
#ax[1].set_zlabel('F1(x)')
#ax[1].set_title('3D surface plot of F1(x)')

# Show the plot
#plt.show()
