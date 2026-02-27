import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    #This is the activation function used for this network. This is the last step of forward pass

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def forward_pass(self, inputs):
        #Takes in inputs, multiplies them with the weights, adds a bias and then returns the 
        #final value returned from the sigmoid activation function
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid(total)
    
#Example run with values
"""weights = np.array([0,1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
print(n.forward_pass(x))"""

class NeuralNetwork:
    """
    A Neural Network with two inputs, a hidden layer with two neurons, and a output layer with one neuron
    Each neuron has the same weights and bias: 
    - w = [0,1]
    - b = 0
    """
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def forward_pass(self, x):
        output_h1 = self.h1.forward_pass(x)
        output_h2 = self.h2.forward_pass(x)

        output_o1 = self.o1.forward_pass(np.array([output_h1, output_h2]))
        return output_o1
    
network = NeuralNetwork()
x = np.array([2, 3])
print(network.forward_pass(x))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))