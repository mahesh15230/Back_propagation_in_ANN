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

def forwardpass(normal_minibatch, no_act_layers, layers, weights, actfunc, minibatchsize, reg_coeff):
	dkn = normal_minibatch[-1]
	normal_minibatch = np.delete(normal_minibatch,np.shape(normal_minibatch)[0]-1,0)
	layers[0][1:,:] = normal_minibatch[:,:minibatchsize]
	no_act_layers = cp.deepcopy(layers)
	for i in range(len(layers) - 1):
		if i<=len(layers)-2:
			no_act_layers[i+1][1:,:] = weights[i]@layers[i]
			layers[i+1][1:,:] = actfunc[i](weights[i]@layers[i],False)
		else:
			no_act_layers[i+1][:,:] = weights[i]@layers[i]
			layers[i+1][:,:] = actfunc[i](weights[i]@layers[i],False)
	error = np.subtract(dkn, actfunc[i+1](layers[-1], False))
	m = 0
	for i in range(len(weights)):
		m += sum(np.shape(weights[i]))
	weight_reg = (reg_coeff / m) * sum([np.sum((weights[i]**2).reshape(np.shape(weights[i]))) for i in range(len(weights))])
	error_energy = error**2 + weight_reg
	return np.sum(error_energy) / (minibatchsize * 2)

def phidashv(layer): # Returns a row matrix
	return np.array([layer.sum(axis=1)]) / np.shape(layer)[1]
	
def localgrad(localgradprev,weight,phidash): #localgradprev is row matrix, so is phidash
	if np.shape(localgradprev)[1] == np.shape(weight)[0]:
		if np.shape(weight)[1] == np.shape(phidash)[1]:
			return phidash * (localgradprev @ weight)
		else:
			print('dim weight and phidash incompatible')
	else:
		print('dim localgrad and weight incompatible')

# Instructions to use the above function
# localgradprev = np.array([[.3,.4,.5]])
# print('localgrad dim', localgradprev)
# weight = np.ones((3,5))
# phidash = np.ones((1,5))
# print('test localgrad',localgrad(localgradprev,weight,phidash))


def backprop(batch_avg_error, no_act_layers, actfunc, weights, layers, learning_param, minibatchsize, reg_coeff):
	phi_dash_vj = actfunc[-1](no_act_layers[-1], True)
	avg_phi_dash = phi_dash_vj.sum(axis=1) / minibatchsize
	local_grad = np.array([batch_avg_error * avg_phi_dash])  # Hardcoded to the assumption output layer has only one neuron
	delta_weights = []
	for i in range(len(layers) - 2, 0, -1):
		print('!!!___i___!!!',i)
		print('dim weight', np.shape(weights[i]))
		print('dim localgrad', np.shape(np.transpose(local_grad)))
		print('dim layer', np.shape(layers[i]))
		print('dim phidashavg', np.shape(phidashv(layers[i])))
		print('type weight update eq',np.shape(np.transpose(local_grad) @ phidashv(layers[i])))
		delta_weights.append(learning_param * (np.transpose(local_grad) @ phidashv(layers[i])) - reg_coeff * abs(weights[i]))
		#local_grad = sum(local_grad @ weights[i]) * layers[i-1].sum(axis = 1).reshape(1, np.shape(layers[i-1])[0]) / np.shape(layers[i-1])[1]
		local_grad = localgrad(local_grad,weights[i],phidashv(layers[i]))





datafile = 'dataset_minibatch_test.txt'
fulldata = np.transpose(np.array(data_init(datafile, full = 1)))
ann_arch = [5,4,3,2,1]
act = [tanh,relu,tanh,relu,logistic]
minibatchsize = 10
learning_param = .001
reg_coeff = .0001
layers = layers_init(ann_arch, minibatchsize)
no_act_layers = layers_init(ann_arch, minibatchsize)
weights = weights_init(ann_arch)
error = forwardpass(fulldata[:minibatchsize,:], no_act_layers, layers, weights, act, minibatchsize, reg_coeff)
print('back prop test', backprop(error, no_act_layers, act, weights, layers, learning_param, minibatchsize, reg_coeff))