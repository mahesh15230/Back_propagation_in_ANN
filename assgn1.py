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
		return testing_data
	elif train != None:
		return training_data
	elif valid != None:
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

def forwardpass(normal_minibatch, no_act_layers, layers, weights, actfunc, minibatchsize, reg_coeff, plot = False):
	dkn = normal_minibatch[-1:,:]
	#print('dim dkn', dkn.shape)
	normal_minibatch = normal_minibatch[:-1,:]
	# print('dim layer', np.shape(layers[0][1:,:]))
	# print('dim minibatch', np.shape(normal_minibatch[:,:minibatchsize]))
	print('dim layer[0]', layers[0].shape)
	print('dim layer[1]', layers[1].shape)
	print('dim layer[2]', layers[2].shape)
	print('dim layer[3]', layers[3].shape)
	layers[0][1:,:] = normal_minibatch[:,:minibatchsize]
	no_act_layers = cp.deepcopy(layers)
	for i in range(len(layers) - 1):
		if i <= len(layers)-2:
			no_act_layers[i+1][1:,:] = weights[i]@layers[i]
			layers[i+1][1:,:] = actfunc[i](weights[i]@layers[i],False)
		else:
			no_act_layers[i+1][:,:] = weights[i]@layers[i]
			layers[i+1][:,:] = actfunc[i](weights[i]@layers[i],False)
	print('dim layer last', layers[-1].shape)
	error = np.subtract(dkn, actfunc[i+1](layers[-1], False))
	m = 0
	for i in range(len(weights)):
		m += sum(np.shape(weights[i]))
	weight_reg = (reg_coeff / m) * sum([np.sum((weights[i]**2).reshape(np.shape(weights[i]))) for i in range(len(weights))])
	error_energy = error**2 + weight_reg
	if plot == True:
		return np.sum(error_energy) / (minibatchsize * 2)
	elif plot == False:
		return np.sum(error) / len(error)

def phidashv(layer): # Returns a row matrix
	return np.array([layer.sum(axis=1)]) / np.shape(layer)[1]
	
def localgrad(localgradprev,weight,phidash): #localgradprev is row matrix, so is phidash
	local_grad = phidash * (localgradprev @ weight)
	return local_grad[:,1:]

# Instructions to use the above function
# localgradprev = np.array([[.3,.4,.5]])
# print('localgrad dim', localgradprev)
# weight = np.ones((3,5))
# phidash = np.ones((1,5))
# print('test localgrad',localgrad(localgradprev,weight,phidash)


def backprop(batch_avg_error, beta, no_act_layers, actfunc, weights, layers, learning_param, minibatchsize, reg_coeff):
	phi_dash_vj = actfunc[-1](no_act_layers[-1], True)
	#print('dim last layer', no_act_layers[-1].shape)
	avg_phi_dash = np.array([phi_dash_vj.sum(axis=1) / minibatchsize])
	#print('dim phidash_avg', avg_phi_dash.shape)
	#print('batch_avg_error', batch_avg_error)
	local_grad = batch_avg_error * avg_phi_dash  # Hardcoded to the assumption output layer has only one neuron
	#print('dim localgrad!!!!!!!!', local_grad.shape)
	del_weight_prev = []
	for i in range(len(layers) - 2, -1, -1):
		# print('!!!___i___!!!',i)
		# print('dim weight', np.shape(weights[i]))
		# print('dim localgrad', np.shape(np.transpose(local_grad)))
		# print('dim layer', np.shape(layers[i]))
		# print('dim phidashavg', np.shape(phidashv(layers[i])))
		del_weight = learning_param * (np.transpose(local_grad) @ phidashv(layers[i])) - reg_coeff * abs(weights[i])		
		del_weight_prev.append(del_weight)
		weights[i][:,:] += beta * del_weight
		local_grad = localgrad(local_grad,weights[i],phidashv(layers[i]))
	del_weight_prev = del_weight_prev[::-1]
	for i in range(len(weights)):
		weights[i][:,:] += (1 - beta) * del_weight_prev[i]

# datafile = 'dataset_minibatch_test.txt'
# fulldata = np.transpose(np.array(data_init(datafile, full = 1)))
# print(fulldata)
# ann_arch = [5,4,3,2,1]
# act = [tanh,relu,tanh,relu,logistic]
# minibatchsize = 10
# learning_param = .001
# reg_coeff = .0001
# layers = layers_init(ann_arch, minibatchsize)
# no_act_layers = layers_init(ann_arch, minibatchsize)
# weights = weights_init(ann_arch)
# error = forwardpass(fulldata[:minibatchsize,:], no_act_layers, layers, weights, act, minibatchsize, reg_coeff, plot = False)
# avg_error = forwardpass(fulldata[:minibatchsize,:], no_act_layers, layers, weights, act, minibatchsize, reg_coeff, plot = True)
# print('ERROR', error)
# print('AVG_ERROR', avg_error)
# print('back prop test', backprop(error, .9, no_act_layers, act, weights, layers, learning_param, minibatchsize, reg_coeff))

def trainingANN(data, epochs, minibatchsize, beta, learning_param, reg_coeff, ann_arch, no_act_layers, layers, actfunc, weights):
	train_error_epoch = []
	valid_error_epoch = []
	for j in range(epochs):
		traindata = np.transpose(data_init(data, train = 1))
		validdata = np.transpose(data_init(data, valid = 1))
		training_error = 0
		valid_error = 0
		for i in range(len(traindata)):
			error = forwardpass(traindata[:,i:i + minibatchsize], no_act_layers, layers, weights, actfunc, minibatchsize, reg_coeff, plot = False)
			training_error += forwardpass(traindata[:,i:i + minibatchsize], no_act_layers, layers, weights, actfunc, minibatchsize, reg_coeff, plot = True)
			backprop(error, beta, no_act_layers, actfunc, weights, layers, learning_param, minibatchsize, reg_coeff)
		training_error /= (len(traindata) / minibatchsize)
		train_error_epoch.append(training_error)
		valid_error += forwardpass(validdata, layers_init(ann_arch, np.shape(validdata)[1]), layers_init(ann_arch, np.shape(validdata)[1]), weights, actfunc, len(validdata), reg_coeff, plot = True)
		valid_error /= len(validdata)
		valid_error_epoch.append(valid_error)
	return train_error_epoch, valid_error_epoch

data = 'dataset_full.txt'
epochs = 3000
minibatchsize = 64
beta = .7
learning_param = .001
reg_coeff = .4
ann_arch = [5,6,6,1]
no_act_layers = layers_init(ann_arch, minibatchsize)
layers = layers_init(ann_arch, minibatchsize)
actfunc = [tanh,relu,tanh,logistic]
weights = weights_init(ann_arch)
__training__, __valid__ = trainingANN(data, epochs, minibatchsize, beta, learning_param, reg_coeff, ann_arch, no_act_layers, layers, actfunc, weights)
