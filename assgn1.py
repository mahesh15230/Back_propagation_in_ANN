import numpy as np
import math

def data_init(datafile, test = None, train = None, valid = None, full = None):
	data = np.loadtxt(datafile)
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

def normalization(training_data,sample_or_label = None):

## Note : This function can take column vector as input only when it's represented like this a = [[1],[2],[3]] and not like this a = [1,2,3]
	data_training = np.transpose(training_data)
	
	for row in range(len(data_training)):

		limmin = min(data_training[row])+.05*abs(min(data_training[row]))
		limmax = max(data_training[row])-.05*abs(max(data_training[row]))
			
		if row != len(data_training) - 1: #Normalization for input neurons : (-1,1)
			data_training[row][:] = [(2*i - limmax - limmin)/(limmax - limmin) for i in data_training[row]]
		else:       #Normalization for output neurons : (0,1)
			data_training[row][:] = [(i - limmin)/(limmax - limmin) for i in data_training[row]]
	
	# x = data_training[:len(data_training) - 1]
	# y = data_training[-1]
	# if sample_or_label == None:
	# 	return np.transpose(x)
	# else:
	# 	return np.transpose(y)
	return np.transpose(data_training)


def weights_init(layersizelist):
	weights = []
	for i in range(len(layersizelist)-1):
		if layersizelist[i+1] == 1:
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[1, layersizelist[i]]))
		else:
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[layersizelist[i+1] - 1, layersizelist[i]])) # layersizelist[i+1] - 1 : minus 1 because weights aren't connected to the bias neuron

	return weights


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

def layers_init(layersizelist, minibatchsize = None):
	if minibatchsize:
		print('mini')
		for i in range(len(layersizelist) - 1):
			layers = np.array([[1] * (layersizelist[i])] * minibatchsize)
		layers.append([[1] * layersizelist[-1]] * minibatchsize)
		layers_ = np.transpose(layers)
		return layers_
	else:
		print('no mini')
		for i in range(len(layersizelist) - 1):
			layers.append(np.transpose([1] * (layersizelist[i])))
		layers.append([1] * layersizelist[-1])
		return layers

def relu(x):
	return x * (x > 0)

def logistic(x):
	return 1/(1 + math.exp(-x))

def relu_derivative(x):
	return (x > 0)

def logistic_derivative(x):
	return logistic(x) * (1 - logistic(x))

def tanh_derivative(x):
	return 1 - math.tanh(x)**2

def activations(layersizelist):
	funclist = []
	for i in range(len(layersizelist) - 2):
		if i%2 == 0:
			funclist.append(math.tanh)
		else:
			funclist.append(relu)
	funclist.append(logistic)
	return funclist


def forwardpass(weights, layers, func, dkn):
	for i in range(len(weights) - 1):
		print('weight dim',[len(weights[i]),len(weights[i][0])])
		print('layer dim',[len(layers[i]),len(layers[i][0])])
		boomba = np.dot(weights[i],layers[i])
		print(len(boomba),len(boomba[0]))
		print(boomba)
		#print(list(map(func[i], np.matmul(weights[i],layers[i]))))
		#layers[i+1][1:] = list(map(func[i], np.matmul(weights[i],layers[i])))
	return dkn[0] - layers[-1][0]


datafile = 'falula.txt'
fulldata = data_init(datafile, full = 1)
print(fulldata)
normal_Data = normalization(fulldata)

# layersizelist = [4,8,4,1]
# weights = weights_init(layersizelist)
# layers = layer_init(layersizelist,5)
# funclist = activations(layersizelist)
# forwardpasstest = forwardpass(weights, layers, funclist, y_vec)
# print(forwardpass)
# print(layers)