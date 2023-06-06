def renorm_lineal(fitness, pob, dec, max_fit):
    # Se crea arreglo de index
    indices = list(range(len(fitness)))

    #Se arregla el arreglo de indices de mayor a menor con base en el arreglo1
    indices.sort(key=lambda i: fitness[i], reverse=True)

    #Se usa el arreglo de indices arreglados para sortear los 2 originales
    sorted_fitness = [fitness[i] for i in indices]
    sorted_pob = [pob[i] for i in indices]

    fit_renorm = []
    num = 0

    for k in range(len(pob)):
        if k == 0:
            num = max_fit
        else:
            num = num - dec
        fit_renorm.append(num)

    return fit_renorm, sorted_pob

fitness = [1, 2, 49, 5, 3]
pob = [["a", "mierda"],  [1110100, 11101],  [5.67, 0.2],  [2.71, 8.8],  [8.28, 0.66]]
fit_aux, sorted_pob= renorm_lineal(fitness, pob, 10, 100)
print(sorted_pob)  #cromosomas ordenado
print(fit_aux)