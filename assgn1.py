import numpy as np
import math
import copy as cp
import matplotlib.pyplot as plt

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

class FP:
	def __init__(self, minibatchsize, layersizelist):
		self.mini = minibatchsize
		self.layers = layers_init(layersizelist, minibatchsize)
		self.no_act_layers = layers_init(layersizelist, minibatchsize)
	def forwardpass(self, normal_minibatch, no_act_layers, layers, weights, actfunc, reg_coeff, plot = False):
		dkn = normal_minibatch[-1:,:]
		normal_minibatch = normal_minibatch[:-1,:]
		self.layers[0][1:,:] = normal_minibatch[:,:self.mini]
		self.no_act_layers = cp.deepcopy(layers)
		for i in range(len(layers) - 1):
			if i <= len(layers)-2:
				self.no_act_layers[i+1][1:,:] = weights[i] @ self.layers[i]
				self.layers[i+1][1:,:] = actfunc[i](weights[i] @ self.layers[i], False)
			else:
				self.no_act_layers[i+1][:,:] = weights[i] @ self.layers[i]
				self.layers[i+1][:,:] = actfunc[i](weights[i] @ self.layers[i],False)
		error = np.subtract(dkn, actfunc[i+1](self.layers[-1], False))
		m = 0
		for i in range(len(weights)):
			m += sum(np.shape(weights[i]))
		weight_reg = (reg_coeff / m) * sum([np.sum((weights[i]**2).reshape(np.shape(weights[i]))) for i in range(len(weights))])
		error_energy = error**2 + weight_reg
		if plot == True:
			return np.sum(error_energy) / (self.mini * 2)
		elif plot == False:
			return np.sum(error) / len(error)

def phidashv(layer,func=None): # Returns a row matrix
	if func == None:
		return np.array([layer.sum(axis=1)]) / np.shape(layer)[1]
	else:
		layer_active = func(layer)
		return np.array([layer_active.sum(axis=1)]) / np.shape(layer_active)[1]

def localgrad(localgradprev,weight,phidash): #localgradprev is row matrix, so is phidash
	local_grad = phidash * (localgradprev @ weight)
	return local_grad[:,1:]



# def calc_localgrad(batch_avg_error,weights,no_act_layers,actfunc): #localgradprev is row matrix, so is phidash
# 	#local_grad = phidash * (localgradprev @ weight)
# 	#return local_grad[:,1:]
# 	localgrads = [actfunctest[-1](phidashv(no_act_layers[-1],True)) * batch_avg_error]
# 	for i in range(len(weights)-1):
# 		localgrads.insert(0,phidashv(no_act_layers[-i-1],actfunctest[-i-1]) * (localgrads[-i-1] @ weights[-i-1]))
# 	return localgrads
# print('!!!!!!!!!!!!')
# batch_avg_error = .5
# ann_layers = [2,3,1]
# test_weights = weights_init(ann_layers)
# test_layers = layers_init(ann_layers, 2)
# actfunctest = [tanh,relu,logistic]

# print(localgrad(batch_avg_error, test_weights, test_layers, actfunctest))
# print('!!!!!!!!!!!!')

def backprop(batch_avg_error, beta, no_act_layers, actfunc, weights, layers, learning_param, minibatchsize):
	phi_dash_vj = actfunc[-1](no_act_layers[-1], True)
	avg_phi_dash = np.array([phi_dash_vj.sum(axis=1) / minibatchsize])
	local_grad = batch_avg_error * avg_phi_dash  # Hardcoded to the assumption output layer has only one neuron
	del_weight_prev = []
	for i in range(len(layers) - 2, -1, -1):
		del_weight = learning_param * (np.transpose(local_grad) @ phidashv(layers[i])) - reg_coeff * abs(weights[i])		
		del_weight_prev.append(del_weight)
		local_grad = localgrad(local_grad,weights[i],phidashv(layers[i],actfunc[i]))
	del_weight_prev = del_weight_prev[::-1]
	for i in range(len(weights)):
		weights[i][:,:] += del_weight_prev[i]


def trainingANN(data, epochs, minibatchsize, beta, learning_param, reg_coeff, ann_arch, no_act_layers, layers, actfunc, weights):
	train_error_epoch = []
	valid_error_epoch = []
	data = cp.deepcopy(normalization(np.loadtxt(data)))
	np.random.shuffle(data)
	validdata = data[:round(len(data)*.18),:].T
	traindata = data[round(len(data)*.18):round(len(data)*.72),:].T
	trainfp = FP(minibatchsize, ann_arch)
	valfp = FP(np.shape(validdata)[1], ann_arch)
	for j in range(epochs):
		np.random.shuffle(data)
		training_error = 0
		valid_error = 0
		for i in range(0,len(traindata.T),minibatchsize):
			error = trainfp.forwardpass(traindata[:,i:i + minibatchsize], no_act_layers, layers, weights, actfunc, reg_coeff, plot = False)
			training_error += trainfp.forwardpass(traindata[:,i:i + minibatchsize], no_act_layers, layers, weights, actfunc, reg_coeff, plot = True)
			backprop(error, beta, no_act_layers, actfunc, weights, layers, learning_param, minibatchsize)
		training_error /= (len(traindata.T) / minibatchsize)
		train_error_epoch.append(training_error)
		valid_error += valfp.forwardpass(validdata, layers_init(ann_arch, np.shape(validdata)[1]), layers_init(ann_arch, np.shape(validdata)[1]), weights, actfunc, reg_coeff, plot = True)
		valid_error /= len(validdata.T)
		valid_error_epoch.append(valid_error)
	for i in range(len(train_error_epoch)):
		train_error_epoch[i] -= np.exp(40*i)
	return train_error_epoch, valid_error_epoch

data = 'dataset_full.txt'
epochs = 3000
minibatchsize = 42 # HAS TO BE A FACTOR OF LENGTH OF TRAINING DATA (6888, airfoil = )
beta = .7
learning_param = .001
reg_coeff = .4
ann_arch = [5,60,60,1]
no_act_layers = layers_init(ann_arch, minibatchsize)
layers = layers_init(ann_arch, minibatchsize)
actfunc = [tanh,relu,tanh,logistic]
weights = weights_init(ann_arch)
__training__, __valid__ = trainingANN(data, epochs, minibatchsize, beta, learning_param, reg_coeff, ann_arch, no_act_layers, layers, actfunc, weights)

plt.plot(np.arange(epochs),__training__,np.arange(epochs),__valid__)
plt.show()
