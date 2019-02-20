import numpy as np
import math
import copy as cp

def data_init(datafile, test = None, train = None, valid = None, full = None):
	data = cp.deepcopy(normalization(np.loadtxt(datafile)))
	if full != None:
		return data
	testing_data = data[:round(len(data) * .1)]
	data = data[round(len(data) * .1):len(data) - 1]
	validation_data = data[:len(data) - (round(len(data) * .8))]
	training_data = data[round(len(data) * .2):]
	if test != None:
		print('test')
		return testing_data
	elif train != None:
		print('train')
		return training_data
	elif valid != None:
		print('valid')
		return validation_data

def normalization(data):
	data_training = np.transpose(cp.deepcopy(data))	
	for row in range(len(data_training)):
		limmin = min(data_training[row])
		limmax = max(data_training[row])		
		if row != len(data_training) - 1: #Normalization for input neurons : (-1,1)
			data_training[row][:] = [(1.8 * (i - limmin) / (limmax - limmin)) - .9 for i in data_training[row]]
		else:       #Normalization for output neurons : (0,1)
			data_training[row][:] = [.9 * (i - limmin)/((limmax - limmin)) for i in data_training[row]]
	return np.transpose(data_training)

def weights_init(layersizelist):
	weights = []
	for i in range(len(layersizelist)-1):
		if layersizelist[i+1] == 1:
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[1, layersizelist[i]]))
		else:
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[layersizelist[i+1] - 1, layersizelist[i]]))
	return np.array(weights)

def layers_init(layersizelist, minibatchsize):
	layer = [np.ones((layersizelist[i],minibatchsize)) for i in range(len(layersizelist))]
	return np.array(layer)

def relu(x, isbackward = False):
	if not isbackward:
		return np.array([[i * int(i > 0) for i in x[j]] for j in range(np.shape(x)[0])])
	else:
		return np.array([[int(i > 0) for i in x[j]] for j in range(np.shape(x)[0])])


def logistic(x, isbackward = False):
	if not isbackward:
		return np.array([[1/(1 + math.exp(-i)) for i in x[j]] for j in range(np.shape(x)[0])])
	else:
		return logistic(x,False) * (1 - logistic(x,False))

def tanh(x, isbackward = False):
	if not isbackward:
		return np.tanh(x)
	else:
		return 1 - tanh(x,False)**2

def forwardpass(normal_minibatch, no_act_layers, layers, weights, actfunc, minibatchsize):
	dkn = normal_minibatch[-1]
	normal_minibatch = np.delete(normal_minibatch,np.shape(normal_minibatch)[0]-1,0)
	layers[0][1:,:] = normal_minibatch[:,:minibatchsize]
	no_act_layers = cp.deepcopy(layers)
	for i in range(len(layers) - 1):
		if i<len(layers)-2:
			no_act_layers[i+1][1:,:] = weights[i]@layers[i]
			layers[i+1][1:,:] = actfunc[i](weights[i]@layers[i],False)
		else:
			no_act_layers[i+1][:,:] = weights[i]@layers[i]
			layers[i+1][:,:] = actfunc[i](weights[i]@layers[i],False)
	error = dkn - actfunc[i+1](layers[-1], False)
	error_energy = error**2
	return sum(error_energy) / minibatchsize

def backprop(batch_avg_error, no_act_layers, actfunc, weights, layers, learning_param, minibatchsize):
	phi_dash_vj = actfunc[-1](no_act_layers[-1], True)
	print('phi_dash', phi_dash_vj)
	print('averaged phi_dash', phi_dash_vj.sum(axis=1))
	local_grad = batch_avg_error * actfunc[-1](no_act_layers[-1], True).sum(axis = 1).reshape(1,1) / minibatchsize # Hardcoded to the assumption output layer has only one neuron
	print('dim locgrad', np.shape(local_grad))
	for i in range(len(layers) - 2, 0, -1):
		print('dim layer', np.shape(layers[i]))
		print('layer bav', np.shape(layers[i].sum(axis = 1)))
		print('type weights', [type(weights[i]),np.shape(weights[i])])
		print('type learn_param', type(learning_param))
		print('type local grad', [type(local_grad), np.shape(local_grad)])
		print('dim layers bav', np.shape(layers[i].sum(axis = 1).reshape(1, np.shape(layers[i])[0])))
		weights[i] += learning_param * local_grad @ layers[i].sum(axis = 1).reshape(1, np.shape(layers[i])[0]) / np.shape(layers[i])[1]
		local_grad = sum(local_grad * weights[i]) * layers[i-1].sum(axis = 1).reshape(1, np.shape(layers[i-1])[0]) / np.shape(layers[i-1])[1]

datafile = 'dataset_minibatch_test.txt'
fulldata = np.transpose(np.array(data_init(datafile, full = 1)))
ann_arch = [5,6,1]
act = [tanh,relu,logistic]
minibatchsize = 10
learning_param = .001
layers = layers_init(ann_arch, minibatchsize)
no_act_layers = layers_init(ann_arch, minibatchsize)
weights = weights_init(ann_arch)
error = forwardpass(fulldata[:minibatchsize,:], no_act_layers, layers, weights, act, minibatchsize)
print('back prop test', backprop(error, no_act_layers, act, weights, layers, learning_param, minibatchsize))