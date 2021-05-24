from sklearn.datasets import load_iris

from particle_swarm_optimization_part import *

import numpy as np

# import the dataset
from sklearn.datasets import load_iris

# load the dataset
dataset = load_iris()

# Store features matrix in input_data
Input_data = dataset.data
# Store target vector in output_data
output_data = dataset.target


# define neural net layers
Input_Nodes = 4
Hidden_Nodes = 20
Output_Nodes = 3


# Apply softmax function on logits and return probabilities.
def softmax_activation_function(logits):

    # Logits of each instance for each class.
    exponentials = np.exp(logits)

    #  probability for each class of each instance.
    softmax = exponentials / np.sum(exponentials, axis = 1, keepdims = True)
    return softmax


# Calculate negative likelihood loss
def Negative_Likelihood_loss(probs, output_data):
    # parameter "prob" is the probability of each instance for each class
    # parameter "output_data" is the Integer representation of each class

    num_samples = len(probs)
    correct_logprobs = -np.log(probs[range(num_samples)])     #calcualte negative log
    return np.sum(correct_logprobs) / num_samples    # calculated value of loss


# Performs forward pass during Neural Net training
def forward_pass (Input_data, output_data, W):
    # parameter "Input_data" is input
    # parameter "output_data" is target
    # parameter W is weight

    if isinstance(W, Particle):
        W = W.x         # check instance

    w1 = W[0: Input_Nodes * Hidden_Nodes].reshape((Input_Nodes, Hidden_Nodes))
    b1 = W[Input_Nodes * Hidden_Nodes:(Input_Nodes * Hidden_Nodes) + Hidden_Nodes].reshape((Hidden_Nodes, ))
    w2 = W[(Input_Nodes * Hidden_Nodes) + Hidden_Nodes:(Input_Nodes * Hidden_Nodes) + Hidden_Nodes +
            (Hidden_Nodes * Output_Nodes)].reshape((Hidden_Nodes, Output_Nodes))
    b2 = W[(Input_Nodes * Hidden_Nodes) + Hidden_Nodes + (Hidden_Nodes * Output_Nodes): (Input_Nodes * Hidden_Nodes) +
          Hidden_Nodes + (Hidden_Nodes * Output_Nodes) + Output_Nodes].reshape((Output_Nodes, ))

    z1 = np.dot(Input_data, w1) + b1     # apply dot product and add bias
    a1 = np.tanh(z1)     # calculate hyperbolic tangent

    logits = np.dot(a1, w2) + b2    # prediction without normalization

    normalize = softmax_activation_function(logits)     # prediction after normalization

    probs = normalize

    loss = Negative_Likelihood_loss(probs, output_data)          # errors in the prediction
    return loss


# Perform forward pass during Neural Net test
def predict(Input_data, W):
    # parameter Input_data is "input"
    # parameter W is "weights"

    w1 = W[0: Input_Nodes * Hidden_Nodes].reshape((Input_Nodes, Hidden_Nodes))
    b1 = W[Input_Nodes * Hidden_Nodes:(Input_Nodes * Hidden_Nodes) + Hidden_Nodes].reshape((Hidden_Nodes,))
    w2 = W[(Input_Nodes * Hidden_Nodes) + Hidden_Nodes:(Input_Nodes * Hidden_Nodes) + Hidden_Nodes + \
        (Hidden_Nodes * Output_Nodes)].reshape((Hidden_Nodes, Output_Nodes))
    b2 = W[(Input_Nodes * Hidden_Nodes) + Hidden_Nodes + (Hidden_Nodes * Output_Nodes): (Input_Nodes * \
        Hidden_Nodes) + Hidden_Nodes + (Hidden_Nodes * Output_Nodes) + Output_Nodes].reshape((Output_Nodes,))

    z1 = np.dot(Input_data, w1) + b1         # apply dot product and add bias
    a1 = np.tanh(z1)                # calculate hyperbolic tangent
    logits = np.dot(a1, w2) + b2                    # prediction without normalization

    normalize = softmax_activation_function(logits)  # prediction after normalization
    probs = normalize
    Y_pred = np.argmax(probs, axis=1)       # Returns the indices of the maximum values along  axis.
    return Y_pred                   # Returns predicted classes.


# calculate accuracy
def get_accuracy(output_data, Y_pred):
    # parameter "output_data" is "correct labels"
    # parameter Y_pred is "predicted labels"

    return (output_data == Y_pred).mean           # return  accuracy


if __name__ == "__main__":
    no_solution = 100
    no_dim = (Input_Nodes * Hidden_Nodes) + Hidden_Nodes + (Hidden_Nodes * Output_Nodes) + Output_Nodes

    w_range = (0.0, 1.0)
    lr_range = (0.0, 1.0)
    iw_range = (0.9, 0.9)
    c = (0.5, 0.3)  # c[0] -> cognitive factor, c[1] -> social factor

    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)
    s.optimize(forward_pass, Input_data, output_data, 100, 1000)
    W = s.get_best_solution()

    Y_pred = predict(Input_data, W)
    accuracy = get_accuracy(output_data, Y_pred)
    print("Accuracy %3f" % accuracy)

