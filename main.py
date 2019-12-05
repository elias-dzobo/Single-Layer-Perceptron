import numpy as np

#Initializing a normalization function(sigmoid function)
def sigmoid(x):
	return 1 / (1 * np.exp(-x))


def sig_derivative(x):
	return x * (1-x)
	

training_inputs = np.array([[0,0,1],
							[1,1,1],
							[1,0,1],
							[0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(1):

	input_layer = training_inputs

	outputs = sigmoid(np.dot(input_layer, synaptic_weights))

	error = training_outputs - outputs #calculating error. difference between predicted value and actual value

	adjustment = error * sig_derivative(outputs) #adjust error based on on error weighted derivative

	synaptic_weights += np.dot(input_layer.T, adjustment) #adjust weights   

print('Synaptic weight after training')
print(synaptic_weights)

print('Outputs after training')
print(outputs)

