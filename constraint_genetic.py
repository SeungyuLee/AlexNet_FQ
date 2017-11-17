#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deap import algorithms, base, creator, tools, gp
import multiprocessing as mp 
import random, numpy, datetime
import array
import sg_transfer 

num_fqlayers = 7
min_bitwidth = 3
max_bitwidth = 8

def evalTotalBitwidth(individual):
	return (sum(individual),)

def feasible(individual):
	if sg_transfer.evalAccMax(individual) > 0.75:
		return True
	return False

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b',fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, min_bitwidth, max_bitwidth)
toolbox.register("individual", tools.initRepeat, creator.Individual, \
		toolbox.attr_bool, n=num_fqlayers)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalTotalBitwidth)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 75.0))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=min_bitwidth, up=max_bitwidth, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def proc(ind_queue, result_queue, gpu_id):
	while(True):
		individual = ind_queue.get()
#		result = sg_transfer.evalAccMax(individual, gpu_id)
		result = toolbox.evaluate(individual)
		result_queue.put([individual, result])
		if(ind_queue.empty()): return

def main():
	random.seed(64)
	pop_num = 100
	gen_num = 40

	pop = toolbox.population(n=pop_num)
	CXPB, MUTPB = 0.5, 0.3
	start_time = datetime.datetime.now()

	hof = tools.HallOfFame(gen_num/2)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=gen_num, stats=stats, halloffame=hof, verbose=True)
	
	end_time = datetime.datetime.now()

	# result documentation
	f = open('result.txt', 'w')
	f.write('start time: %s\n' % (str(start_time)))
	f.write('end time: %s\n' % (str(end_time)))

	f.write('population number: %s\n' % (pop_num))
	f.write('generation number: %s\n' % (gen_num))

	f.write(str(log))
	f.write('\n')
	for i in range(len(pop)):
		f.write('layers: %s\n' %(pop[i]))	
	for i in range(len(hof)):
		f.write('layers: %s\n' %(hof[i]))	

	f.close()

	return 

if __name__ == "__main__":
	main()
