#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./caffe-FQ/python')

from caffe.proto import caffe_pb2
import caffe
import numpy as np
import random

import time
import json
from collections import OrderedDict
"""
base_proto = "./lenet_train_test.prototxt"
base_weight = "./lenet_iter_10000.caffemodel"
fq_proto = "./lenet_train_test_fq.prototxt"
fq_weight = "./lenet_fq.caffemodel"
"""
# basic functions
def setParam(net, layer_name, index, data):
	layer_idx = list(net._layer_names).index(layer_name)
	np.copyto(net.layers[layer_idx].blobs[index].data, data)

def getBaseParam(proto, weight):
	net = caffe.Net(proto, weight, caffe_pb2.TEST)
	rtn_dict = {}
	for idx, name in enumerate(net._layer_names):
		lst = []
		for blob in net.layers[idx].blobs:
			lst.append(blob.data)
		if len(lst) > 0:
			rtn_dict[name] = lst
	return rtn_dict

def evalAccMax(individual, gpu_id=0):
	import sys
	sys.path.append('./caffe-FQ/python')
	from caffe.proto import caffe_pb2
	import caffe
	import time
	import json
	base_proto = "./convnet_train_test.prototxt"
	base_weight = "./convnet_full_iter_60000.caffemodel.h5"
	fq_proto = "./convnet_train_test_fq.prototxt"
	fq_weight = "./convnet_fq.caffemodel"

	# load trained parameters
	param_dict = getBaseParam(base_proto, base_weight)

	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)
	
	net = caffe.Net(fq_proto, caffe_pb2.TRAIN)

	ind_idx = 0
	for idx, name in enumerate(net._layer_names):
		if name in param_dict:
			for b_idx, blob in enumerate(param_dict[name]):
				setParam(net, name, b_idx, blob)

			if net.layers[idx].type == "FQConvolution" or \
				net.layers[idx].type == "FQInnerProduct":
				setParam(net, name, len(param_dict[name]), \
						[2**individual[ind_idx], 0, 0, 0])
				ind_idx+=1

		if net.layers[idx].type == "FQActive":
			setParam(net, name, 0, [2**individual[ind_idx], 0, 0, 0])
			ind_idx+=1

	net.save(fq_weight)
	net = caffe.Net(fq_proto, fq_weight, caffe_pb2.TEST)
	
	acc = 0
	batches_num = 100
	test_start_time = time.time()	
	for idx in range(batches_num):
		acc += net.forward()['accuracy']
	test_end_time = time.time()

	acc = acc / batches_num
	elapsed_time = test_end_time - test_start_time
	
	# result documentation
	data = OrderedDict()
	data['bit width of layers'] = str(individual)
	data['accuracy'] = acc
	data['elapsed'] = elapsed_time

	f = open('verbose_result.json', "a")
	json.dump(data, f)
	f.write("\n")
	f.close()

	return acc,
