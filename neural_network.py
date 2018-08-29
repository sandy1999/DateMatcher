# numpy functions 
from numpy import dot, array, exp, random

# making a Neural Network Class 
class NeuralNetwork(object):
    # init function or constructor
    def __init__(self, input_layer = 3, hidden_layer_1= 5, hidden_layer_2= 4 , output_layer=6, learning_rate=0.01):

        # stating a random seed 
        random.seed(1)

        # defining weights
        self.synaptic_weight1 = 2 * random.random((input_layer, hidden_layer_1)) - 1
        self.synaptic_weight2 = 2 * random.random((hidden_layer_1, hidden_layer_2)) - 1
        self.synaptic_weight3 = 2 * random.random((hidden_layer_2, output_layer)) - 1
        self.learning_rate = learning_rate
    # sidmoid function
    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x))

    # derivative of sigmoid function 
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_input_set, training_output_set, num_iterations):

        # train in num iterations
        for i in range(num_iterations):
            activation_layer_2 = self.__sigmoid(dot(training_input_set, self.synaptic_weight1))
            activation_layer_3 = self.__sigmoid(dot(activation_layer_2, self.synaptic_weight2))
            output = self.__sigmoid(dot(activation_layer_3, self.synaptic_weight3))

            # error funtion 
            del4 = (training_output_set - output) * self.__sigmoid(output)

            # error function for rest of our layers
            del3 = dot(self.synaptic_weight3, del4.T) * (self.__sigmoid_derivative(activation_layer_3).T)
            del2 = dot(self.synaptic_weight2, del3) * (self.__sigmoid_derivative(activation_layer_2).T)

            # getting ajustment values 
            adjustment3 = dot(activation_layer_3.T, del4)
            adjustment2 = dot(activation_layer_2.T, del3.T)
            adjustment1 = dot(training_input_set.T, del2.T)

            # adjusting weights based on these adjustments
            self.synaptic_weight1 += (self.learning_rate * adjustment1)
            self.synaptic_weight2 += (self.learning_rate * adjustment2)
            self.synaptic_weight3 += (self.learning_rate * adjustment3)

    # predict function to predict values of our features
    def predict(self, input_values):
        activation_layer_2 = self.__sigmoid(dot(input_values, self.synaptic_weight1))
        activation_layer_3 = self.__sigmoid(dot(activation_layer_2, self.synaptic_weight2))
        output = self.__sigmoid(dot(activation_layer_3, self.synaptic_weight3))

        return output

def main():
    nn = NeuralNetwork(1, output_layer=1)

    nn.train(array([[1]]), array([[0]]), 1000)
    print(nn.predict(array([[1]])))

if __name__ == '__main__':
    main()