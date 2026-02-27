- The Neuron in a Neural Networks takes inputs, multiplies them with weights and adds a bias. It then feeds it into an activation function. The Sigmoid activation function was used in this project. The Sigmoid function takes in inputs in the negative infinity to positive infinity range and compresses them to return a number in the (0,1) range. This means that really small numbers would be closer to 0 and really large numbers would be closer to 1.

- This process is known as forward pass. This uses a dot product to multiply the input values and the weights, add all of them and then add a bias to this. This value is then fed into the Sigmoid function to give us our output value. 

- A hidden layer is any layer between the input (first) layer and the output (last) layer. There can be multiple
- The way the neutral network will work is that the inputs for the hidden layer neurons will be the outputs from input layer's forward pass method. Then running forward pass again with the hidden layers values as the inputs will be the final output.

- Calculating loss is the way of quantifying how good our data really is. This way we can decide how much better the network needs to perform
- We used MSE: Mean Squared Error here. 
    - Main idea is to minimize the loss, meaning that we got better predictions. Total point of training neural networks is trying to minimize its loss
    - The formula for MSE is: $${MSE = {\frac{1/n}}\Sigma}$$

- After being able to calculate the MSE, we know if we need to change our weights and biases. But the way to do so so that the loss decreases is to use gradients. The way to do this is to evaluate the gradients in the Loss Function. The Loss function contains all the weights and biases and the gradients will allow us to find out how to reduce these values so that we can decrease the loss to its maximum.

- The Gradient for example, weight 1 or $${w_1}$$, is the partial derivative of L with respect to $${y_{pred}}$$ multiplied by the partial derivative of $${y_{pred}}$$ with respect to $${h_1}$$, because thats the only hidden layer that $${w_1}$$ affects, which is then multiplied by the partial derivative of $${h_1}$$ with respect to $${w_1}$$. This is backpropagation as we're working from the end of the Loss function. **FINISH LATER**