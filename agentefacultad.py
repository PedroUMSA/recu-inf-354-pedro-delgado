import numpy as np
import random
import array
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


distancias=[[   0,  555, 1945,  575,  792, 1274, 1620,   96,  483, 1318,  829,  901,],
            [ 555,    0,  560, 1200, 1209,  209, 1544, 1598, 1551, 1763, 1337, 1123,],
            [1945,  560,    0,  644,  791,  669,  776,   59, 1378,  351, 1425,  686,],
            [ 575, 1200,  644,    0, 1568,  108,  855,  705, 1454,  112,  233,  949,],
            [ 792, 1209,  791, 1568,    0, 1064,  745, 1897, 1399, 1512, 1807,  205,],
            [1274,  209,  669,  108, 1064,    0, 1201,  460,   94,  350,  110, 1815,],
            [1620, 1544,  776,  855,  745, 1201,    0, 1477,  495, 1108,  727,  759,],
            [  96, 1598,   59,  705, 1897,  460, 1477,    0,  371,  673, 1543, 1756,],
            [ 483, 1551, 1378, 1454, 1399,   94,  495,  371,    0, 1223,  242, 1409,],
            [1318, 1763,  351,  112, 1512,  350, 1108,  673, 1223,    0, 1761, 1693,],
            [ 829, 1337, 1425,  233, 1807,  110,  727, 1543,  242, 1761,    0,  981,],
            [ 901, 1123,  686,  949,  205, 1815,  759, 1756, 1409, 1693,  981,    0,]]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()


toolbox.register("indices", random.sample, range(12), 12)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def totalRuta(individual):
    distanciaGen = distancias[individual[-1]][individual[0]]
    print(individual)
    print(individual[0:-1])
    print(individual[1:])
    for ruta1, ruta2 in zip(individual[0:-1], individual[1:]):
        distanciaGen += distancias[ruta1][ruta2]
    
    return distanciaGen,

#definimos una cruzamiento ordenado
toolbox.register("mate", tools.cxOrdered)
#definimos una mutacion para atributos que tienen una secuencia
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
#definimos la seleccion para el mejor individuo elegido al azar de un conjunto de tamaño k
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", totalRuta)

def main():
    random.seed(200)
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 200, stats=stats, halloffame=hof, verbose=False)
    
    return pop, stats, hof


    
pop,stats, hof= main()

data = hof[0]
print('Ruta:\n')
print('costo:',totalRuta(data))
for i in data:
    if i==0:
        print(i,': agronomia')
    if i==1:
        print(i,': arquitectura')
    if i==2:
        print(i,': economica')
    if i==3:
        print(i,': geológicas')
    if i==4:
        print(i,': ciencias puras')
    if i==5:
        print(i,': ciencias sociales')
    if i==6:
        print(i,': derecho')
    if i==7:
        print(i,': humanidades')
    if i==8:
        print(i,': ingeneria')
    if i==9:
        print(i,': medicina')
    if i==10:
        print(i,': odontologia')
    if i==11:
        print(i,': tecnologica')