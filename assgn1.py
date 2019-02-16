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

## Note : This function can take column vector as input only when it's represented like this a = [[1],[2],[3]] and not like this a = [1,2,3]
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
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[layersizelist[i+1] - 1, layersizelist[i]])) # layersizelist[i+1] - 1 : minus 1 because weights aren't connected to the bias neuron

	return np.array(weights)


# def layer_init(layersizelist, inputlayer, minibatchsize = None): #Assuming input layer is a normal list
# 	layers = []
# 	for i in range(len(layersizelist) - 1):
# 		layers.append(np.transpose([1] + [None] * (layersizelist[i] - 1)))
# 	layers.append([None] * layersizelist[-1])
# 	if minibatchsize == None:
# 		#layers[0][1:] = 
# 		return layers
# 	# else:
# 	# 	for i in range(len(layersizelist)):
# 	# 		# mini batches.....


# def layers_init(layersizelist, minibatchsize = None):
# 	if minibatchsize:
# 		print('mini')
# 		for i in range(len(layersizelist)):
# 			layers = [[1] * (layersizelist[i])] * minibatchsize
# 		return np.transpose(layers)
# 	else:
# 		layers = []
# 		print('no mini')
# 		for i in range(len(layersizelist)):
# 			layers.append(cp.deepcopy(np.transpose([1] * (layersizelist[i]))))
# 		return np.array(layers)

def layers_init(layersizelist, minibatchsize):
	layer = [np.ones((layersizelist[i],minibatchsize)) for i in range(len(layersizelist))]
	return np.array(layer)

def relu(x, isnotforward = False):
	if not isnotforward:
		return np.array([[i * int(i > 0) for i in x[j]] for j in range(np.shape(x)[0])])
	else:
		return np.array([[int(i > 0) for i in x[j]] for j in range(np.shape(x)[0])])

temp = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
print('relu forw',relu(temp,False))
print('relu back', relu(temp,True))

def logistic(x, isnotforward = False):
	if not isnotforward:
		return np.array([[1/(1 + math.exp(-i)) for i in x[j]] for j in range(np.shape(x)[0])])
	else:
		return logistic(x,False) * (1 - logistic(x,False))

print('logistic forw', logistic(temp,False))
print('logistic back', logistic(temp,True))



def tanh(x, isnotforward = False):
	if not isnotforward:
		return np.tanh(x)
	else:
		return np.array([[1 - np.tanh(i)**2 for i in x[j]] for j in range(np.shape(x)[0])])

print('tanh forw', tanh(temp,False))
print('tanh back', tanh(temp,True))


def activations(layersizelist):
	funclist = []
	for i in range(len(layersizelist) - 2):
		if i%2 == 0:
			funclist.append(np.tanh)
		else:
			funclist.append(relu)
	funclist.append(logistic)
	return funclist


# def forwardpass(weights, layers, func, dkn):
# 	for i in range(len(weights) - 1):
# 		print('weight dim',[len(weights[i]),len(weights[i][0])])
# 		print('layer dim',[len(layers[i]),len(layers[i][0])])
# 		boomba = np.dot(weights[i],layers[i])
# 		print(len(boomba),len(boomba[0]))
# 		print(boomba)
# 		#print(list(map(func[i], np.matmul(weights[i],layers[i]))))
# 		#layers[i+1][1:] = list(map(func[i], np.matmul(weights[i],layers[i])))
# 	return dkn[0] - layers[-1][0]

def forwardpass(normal_minibatch, layers, weights, actfunc, minibatchsize):
	dkn = np.transpose(cp.deepcopy([normal_minibatch[-1] for i in range(minibatchsize)]))
	print('old dim normal batch', np.shape(normal_minibatch))
	print('old minibatch', normal_minibatch)
	normal_minibatch = np.delete(normal_minibatch,np.shape(normal_minibatch)[0]-1,0)
	print('new dim normal batch', np.shape(normal_minibatch))
	print('new layer', normal_minibatch)
	layers[0] = normal_minibatch[:,:minibatchsize]
	for i in range(len(layers)):
		print('dim weights',np.shape(weights[i]))
		print('dim layer', np.shape(layers[i]))
		print('dim dot', np.shape(weights[i]@layers[i]))
		layers[i][1:,:] = np.apply_along_axis(actfunc[i],0,np.dot(weights[i],layers[i]))
		print('layer[index]', layers[i])
	print('dim layer[-1]', np.shape(layers[-1]))
	print('dim dkn', np.shape(dkn))


	# for i in range(len(normal_data)):
	# 	layers[i][1:,:] = np.dot(weights[i], normal_data[:,i:i + minibatchsize])

datafile = 'dataset_minibatch_test.txt'
fulldata = np.transpose(np.array(data_init(datafile, full = 1)))
ann_arch = [4,6,6,6,1]
minibatchsize = 10
layers = layers_init(ann_arch, minibatchsize)
weights = weights_init(ann_arch)
act_func = activations(ann_arch)
np.savetxt('normal__.txt',np.transpose(fulldata[:minibatchsize,:]))
forwardpass(fulldata[:minibatchsize,:], layers, weights, act_func, minibatchsize)

# layersizelist = [4,8,4,1]
# weights = weights_init(layersizelist)
# layers = layer_init(layersizelist,5)
# funclist = activations(layersizelist)
# forwardpasstest = forwardpass(weights, layers, funclist, y_vec)
# print(forwardpass)
# print(layers)