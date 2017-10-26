import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T

#TO-DO
# Create policy networks
# Find gradient of loss function

layer_dict = {
                1: "Conv2D",
                2: "MaxPool"
}

def makeVggNet():
    return models.vgg16()

def getLayerInfo(model):
    features = []
    for feature in model.features:
        if (str(feature).startswith('Conv')):
            features.append((1, feature.kernel_size[0], feature.stride[0], feature.padding[0], feature.out_channels))
    return features, torch.Tensor(features)

class LayerRemovalPolicyNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=5, output_size=2):
        super(LayerRemovalPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.hidden = self.init_hidden_weights(hidden_size, 1, 2)
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print(self.hidden)
        m = self.hidden
        output, self.hidden = self.lstm(x, (self.hidden, self.hidden))         # WHAT DOES THIS SELF.LSTM DO? WY DOES IT RETURN TWO THINGS?
        print(self.hidden)
        self.hidden = m
        # output = self.fc(output.view(-1, output.size(2)*2))
        # print(output.view(-1, output.size(2)).size())
        # print(output.size())
        output = nn.Softmax()(output.view(-1, output.size(2)*2))
        return output

    def init_hidden_weights(self, hidden_size, batch_size=1, num_directions=1):
        return Variable(torch.rand(num_directions, batch_size, hidden_size))

l = LayerRemovalPolicyNetwork()
model = models.vgg16()
_, features = getLayerInfo(model)
l(Variable(features[0]).view(1,-1))

# import torch
# import torchvision
# import torch.nn as nn
# from torch.autograd import Variable
#
#
# rnn = nn.LSTM(10, 20, bidirectional=True)
# input = Variable(torch.randn(1, 10))
# h0 = Variable(torch.randn(1*2, 1, 20))
# c0 = Variable(torch.randn(1*2, 1, 20))
# output, hn = rnn(input, (h0, c0))
#
# rnn = nn.LSTM(10, 20, 2)
# input = Variable(torch.randn(5, 3, 10))
# h0 = Variable(torch.randn(2, 3, 20))
# c0 = Variable(torch.randn(2, 3, 20))
# output, hn = rnn(input, (h0, c0))
#

# yukezhu tensorflow reinforce
# pytorch reinforce
# What are we trying  to do:
# We're given a network.
# QUESTION: How do we get information of its layers and size of network?
#     By printing model and parsing the resultant string
#     (GetNetworkInfo)
# Apply the layer pruning Policy
#     Layer Pruning Policy
#         For N iterations:
#             For each layer:
#                 Take an action to remove or keep the layer.
#                 QUESTION: Does the LSTM network correctly predict whether or not to remove?
#                 QUESTION: How do we actually remove or keep a layer
#             QUESTION: Do I train the network now or just directly use it to try an predict output
#             Calculate the reward function after that iteration
#                 Find the accuracy of the student network
#                 Find the number of parameters for the student network
#             Use the policy gradient technique to update the weights
#             QUESTION: What are the parameters for the gradient
#             QUESTION: Where do we use the reward function value in the policy gradient part?
#             QUESTION: What is the form of the expression for the gradients
#             QUESTION: Is theta updated by adding gradient or by changing to gradient
#             QUESTION: How do we back propogate the error to the LSTM network
#             QUESTION: How to find the stage one candidate?
#
#             QUESTION: Do we keep an LSTM network and separate Policy Algorithm?


# List of things done:
#
# What is the problem (a brief description).
# What is the status in solving the problem?
# What are the current intermediate results.
# What is the plan for completion.
#
# Problem statement:
# Problem statement:
#     Implement the N2N compresssion network that uses policy gradient approach to find an optimal policy that can compress the given network.
#
# Current status:
#
# - Read and understood the N2N paper
#     Algorithm briefing
#     The compression network is divided into two stages - layer removal and layer shrinkage each implementing a separate policy.
#     The policy network is formulated as a Markov Decision Process.
#     The compression policy algorithm proceeds as follows
#         s0 = teacher N/W:
#             repeat for n iterations:
#                 repeat for timesteps t in range(layers):
#                     (at = Policy(s t-1, theta)) //choose an action according to policy
#                     st = T(s t-1, at) //Transition to new state
#                 R = r(sL) //Calculate reward at stage
#                 Theta += Gradient J(theta) //Update the parameters of the policy network
#
#     In order to implement the policy we used a bi-directional LSTM that takes in as input a sequence of layer information and
#     output probability of keeping the layer vs removing. The layer information is the tuple(layer type, kernel_size, stride, padding, output_size)
# - Implemented function to extract information of the layer and store as list
# - Implemented LSTM network to stochastically map layer information to remove/keep action
# - Implemented the transistion function to remove/keep layer
#
# Plan for completion:
# We will complete the layer shrinkage policy, add it to the layer removal section and complete the policy network.
# We will run tests on the final the policy network.
#
#
#
#
#
#




# ''
