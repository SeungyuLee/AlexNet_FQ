#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deap import algorithms, base, creator, tools, gp
import multiprocessing as mp 
import random, numpy, datetime

import sg_transfer 

num_fqlayers = 7
min_bitwidth = 3
max_bitwidth = 8

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, min_bitwidth, max_bitwidth)
toolbox.register("individual", tools.initRepeat, creator.Individual, \
		toolbox.attr_bool, n=num_fqlayers)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", sg_transfer.evalAccMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=min_bitwidth, up=max_bitwidth, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def proc(ind_queue, result_queue, gpu_id):
	while(True):
		individual = ind_queue.get()
#		result = sg_transfer.evalAccMax(individual, gpu_id)
		result = toolbox.evaluate(individual, gpu_id)
		result_queue.put([individual, result])
		if(ind_queue.empty()): return

def main():
	random.seed(64)
	pop_num = 80
	gen_num = 20

	pop = toolbox.population(n=pop_num)
	CXPB, MUTPB = 0.5, 0.3

	print("Start of evolution")
	
	fitnesses = [None] * pop_num
	
	ind_q = mp.Queue(pop_num)
	result_q = mp.Queue(pop_num)
	for ind in range(len(pop)):
		ind_q.put(pop[ind])

	process_gpu = [mp.Process(target=proc, args=(ind_q,result_q,0,)), mp.Process(target=proc, args=(ind_q,result_q,1,))]
	process_gpu[0].start()
	process_gpu[1].start()
	process_gpu[0].join()
	process_gpu[1].join()

	ind_res_tuple = [(None, None)] * pop_num
	
	for ind in range(len(pop)):
		ind_res_tuple[ind] = result_q.get()

	# it is observed that the order of 'pop' and 'ind_res_tuple' are the same
	for ind in range(len(pop)):
		pop[ind] = ind_res_tuple[ind][0]
		fitnesses[ind] = ind_res_tuple[ind][1]

	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	print(" Evaluated %i individuals" % len(pop))

	fits = [ind.fitness.values[0] for ind in pop]

	g = 0
	
	start_time = datetime.datetime.now()
	while max(fits) < 100 and g < gen_num:
		g = g + 1
		print("-- Generation %i --" % g)

		offspring = toolbox.select(pop, len(pop))
		offspring = list(map(toolbox.clone, offspring))
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values

		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		# multiprocessing again
		ind_q = mp.Queue(len(invalid_ind))
		result_q = mp.Queue(len(invalid_ind))
		for i in range(len(invalid_ind)):
			ind_q.put(invalid_ind[i])

		process_gpu = [mp.Process(target=proc, args=(ind_q,result_q,0,)), mp.Process(target=proc, args=(ind_q,result_q,1,))]
		process_gpu[0].start()
		process_gpu[1].start()
		process_gpu[0].join()
		process_gpu[1].join()

		ind_res_tuple = [(None, None)] * len(invalid_ind)
		fitnesses = [None] * len(invalid_ind)	

		print ("result_q.get() start")
		for ind in range(len(invalid_ind)):
			ind_res_tuple[ind] = result_q.get()

		print ("fitnesses[ind] start")
		for ind in range(len(invalid_ind)):
			fitnesses[ind] = ind_res_tuple[ind][1]

		print ("ind.fitness.values = fit start")
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		print(" Evaluated %i individuals" % len(invalid_ind))

		pop[:] = offspring

		fits = [ind.fitness.values[0] for ind in pop]

		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5

		print(" Min %s" % min(fits))
		print(" Max %s" % max(fits))
		print(" Avg %s" % mean)
		print(" Std %s" % std)
		
	end_time = datetime.datetime.now()
	print("-- End of evolution --")

	best_ind = tools.selBest(pop, pop_num)
	for i in range(len(best_ind)):
		print("Best individual %s is %s, %s" % (i, best_ind[i], best_ind[i].fitness.values))

	# result documentation
	f = open('result.txt', 'w')
	f.write('start time: %s\n' % (str(start_time)))
	f.write('end time: %s\n' % (str(end_time)))

	f.write('population number: %s\n' % (pop_num))
	f.write('generation number: %s\n' % (gen_num))

	for i in range(len(best_ind)):
		f.write('layers: %s, accuracy: %s\n' %(best_ind[i], best_ind[i].fitness.values))
	f.close()

	return 

if __name__ == "__main__":
	main()
