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
    """
    """
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

    - Manually coded just for learning

    - The dataset for this code is people weights and heights used to predict their gender
    """

    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def forwardpass(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues): 
        #data: [n,2] array, n is number of samples in dataset
        #all_y_trues is numpy array with n elements
        learning_rate = 0.1
        epochs = 1000 #number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * x[0] + self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                #Partial Derivatives

                d_L_d_y_pred = -2 * (y_true - y_pred)

                #Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                #Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #Updating weights and biases
                self.w1 -= learning_rate * d_L_d_y_pred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_L_d_y_pred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_L_d_y_pred * d_ypred_d_h1 * d_h1_d_b1

                self.w3 -= learning_rate * d_L_d_y_pred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_L_d_y_pred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_L_d_y_pred * d_ypred_d_h2 * d_h2_d_b2

                self.w5 -= learning_rate * d_L_d_y_pred * d_ypred_d_w5
                self.w6 -= learning_rate * d_L_d_y_pred * d_ypred_d_w6
                self.b3 -= learning_rate * d_L_d_y_pred * d_ypred_d_b3
            if epochs % 10 == 0:
                y_preds = np.apply_along_axis(self.forwardpass, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
"""
network = NeuralNetwork()
x = np.array([2, 3])
print(network.forward_pass(x))
"""    
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
"""
y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))
"""

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

data = np.array([
    [-2,-1], #Alice
    [25, 6], #Bob
    [17, 4], #Charlie 
    [-15, -6], #Diana
])
all_y_trues = np.array([
    1, #Alice
    0, #Bob
    0, #Charlie
    1, #Diana
])

network = NeuralNetwork()
network.train(data, all_y_trues)

emily = np.array([-7, -3]) #128 pounds, 63 inches
frank = np.array([20, 2]) #150 pounds, 68 inches
print("Emily: %.3f" % network.forwardpass(emily))
print("Frank: %.3f" % network.forwardpass(frank))