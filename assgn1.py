import numpy as np
import math




# 1. Test correctness of normalization function with falula by normalizing and denormalizing
# 2. Test the layer_init function
# 3. Rewrite forwardpass function if required



datafile = 'falula.txt'
data = np.transpose(np.loadtxt(datafile))

def normalization(datafile,sample_or_label = None):

## Note : This function can take column vector as input only when it's represented like this a = [[1],[2],[3]] and not like this a = [1,2,3]
	
	training_data = np.loadtxt(datafile)
	data_training = np.transpose(training_data)
	
	for row in range(len(data_training)):

		limmin = min(data_training[row])+.05*abs(min(data_training[row]))
		limmax = max(data_training[row])-.05*abs(max(data_training[row]))
			
		if row != len(data_training) - 1: #Normalization for input neurons : (-1,1)
			data_training[row][:] = [(2*i - limmax - limmin)/(limmax - limmin) for i in data_training[row]]
			np.savetxt('normalized falula.txt',np.transpose(data_training))
		else:       #Normalization for output neurons : (0,1)
			data_training[row][:] = [(i - limmin)/(limmax - limmin) for i in data_training[row]]
			np.savetxt('normalized falula.txt',np.transpose(data_training))
	
	x = data_training[:len(data_training) - 1]
	y = data_training[-1]
	if sample_or_label == None:
		return np.transpose(x)
	else:
		return np.transpose(y)


def weights_init(layersizelist):
	weights = []
	for i in range(len(layersizelist)-1):
		if layersizelist[i+1] == 1:
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[1, layersizelist[i]]))
		else:
			weights.append((1/math.sqrt(layersizelist[i]))*np.random.uniform(-1,1,[layersizelist[i+1] - 1, layersizelist[i]])) # layersizelist[i+1] - 1 : minus 1 because weights aren't connected to the bias neuron

	return weights


def layer_init(layersizelist): #Assuming input layer is a normal list
	layers = []
	for i in range(len(layersizelist) - 1):
		layers.append(np.transpose([1] + [None] * (layersizelist[i] - 1)))
	layers.append([None] * layersizelist[-1])
	return layers


def relu(x):
	if x <= 0:
		return 0
	else:
		return x

def logistic(x):
	return 1/(1 + math.exp(-x))

def activations(layersizelist):
	funclist = []
	for i in range(len(layersizelist) - 2):
		if i%2 == 0:
			funclist.append(math.tanh)
		else:
			funclist.append(relu)
	funclist.append(logistic)
	return funclist


def forwardpass(weights, layers, func, data):
	for i in range(len(weights) - 1):
		layers[i+1][1:] = list(map(func[i], np.matmul(weights[i],layers[i])))



weights = [[[1,0],[0,1]],[1,1,1]]
training_data = normalization(datafile)
layers = layer_init([2,3,1])
funclist = activations([2,3,1])
print(funclist)
