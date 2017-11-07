import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as transforms
import collections
import copy

#TO-DO
# Create policy networks
# Find gradient of loss function

batch_size = 100
num_epochs = 2
learning_rate = 0.01
num_iterations = 10

# layer_dict = {
#                 1: "Conv2D",
#                 2: "MaxPool"
# }

train_dataset = datasets.CIFAR10(root='./CIFAR_data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./CIFAR_data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def makeVggNet():
    return models.vgg16()

'''
Goes through the layers of the model using its feature dictionary.
It stores the information of each layer in an ordered dictionary and returns the value.
'''
def getLayerInfo(model):
    features = collections.OrderedDict()
    featureDictionary = model._modules
    for index in featureDictionary:
        feature = featureDictionary[index]
        if (str(feature).startswith('Conv') and feature.in_channels == feature.out_channels):
            features[index] = (1, feature.kernel_size[0], feature.stride[0], feature.padding[0], feature.out_channels)
    return features

'''
Creates a layer removal policy network using a bi-directional LSTM
It outputs the probability of removing vs keeping a layer
'''
class LayerRemovalPolicyNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=5, output_size=2):
        super(LayerRemovalPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.hidden = self.init_hidden_weights(hidden_size, 1, 2)
        self.cell_state = self.init_hidden_weights(hidden_size, 1, 2)

    def forward(self, x):
        output, (self.hidden, self.cell_state) = self.lstm(x, (self.hidden, self.cell_state))         # WHAT DOES THIS SELF.LSTM DO? WY DOES IT RETURN TWO THINGS?
        output = self.fc(output.view(-1, output.size(2)*2))
        output = nn.Softmax()(output)
        return output

    def init_hidden_weights(self, hidden_size, batch_size=1, num_directions=1):
        return Variable(torch.rand(num_directions, batch_size, hidden_size))

'''
Create an exact copy of the teacher network - student network.
Go through the layers of the student network one by one.
Prune the layers based on decision taken by the LayerRemovalPolicyNetwork
Return student network
'''
def pruneLayers(model):
    student_model = copy.deepcopy(model)
    model_features = getLayerInfo(student_model)
    l = LayerRemovalPolicyNetwork()
    for index in model_features:
        feature = model_features[index]
        if (l(Variable(torch.Tensor(feature)).view(1,-1)))[0][0].data[0] > 0.5:
            del(student_model._modules[index])
    return student_model

'''
Train the student network using the logits of the teacher network
'''

def mse_loss(input, target):
    return torch.sum((input.data - target.data)**2) / input.data.nelement()

def train(teacher_model, student_model):
    criterion = nn.MSELoss()
    # criterion = mse_loss
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            teacher_outputs = teacher_model(images)
            teacher_outputs = Variable(teacher_outputs.data)

            optimizer.zero_grad()
            student_outputs = student_model(images)
            print(student_outputs.size())
            error = criterion(student_outputs, teacher_outputs)
            error.backward()
            optimizer.step()
            if (i+1) % (1000/batch_size) == 0:
                print (error.data[0])
            print("Epoch Complete")
    return student_model

def test(model):
    correct = 0
    total = 0

    for images, labels in test_loader:
	images = Variable(images).cuda()

        #if(use_gpu):
         #   images = images.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #print(predicted.cpu())
        correct += (predicted.cpu() == labels.cpu()).sum()
	print(correct, total)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

input_size=32

class CustomResnet(nn.Module): # Extend PyTorch's Module class
	def __init__(self, num_classes = 10):
		super(CustomResnet, self).__init__() # Must call super __init__()

		# Define the layers of the network here
		# There should be 17 total layers as evident from the diagram
		# The parameters and names for the layers are provided in the diagram
		# The variable names have to be the same as the ones in the diagram
		# Otherwise, the weights will not load

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.lyr1conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
		self.lyr1bn1 = nn.BatchNorm2d(64)
		self.lyr1relu1 = nn.ReLU(inplace=True)
		self.lyr1conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
		self.lyr1bn2 = nn.BatchNorm2d(64)
		self.lyr1relu2 = nn.ReLU(inplace=True)
		self.lyr2conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
		self.lyr2bn1 = nn.BatchNorm2d(64)
		self.lyr2relu1 = nn.ReLU(inplace=True)
		self.lyr2conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
		self.lyr2bn2 = nn.BatchNorm2d(64)
		self.lyr2relu2 = nn.ReLU(inplace=True)
		self.lyr2conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
		self.lyr2bn3 = nn.BatchNorm2d(1)
		self.lyr2relu3 = nn.ReLU(inplace=True)
		self.fc = nn.Linear(int(input_size*input_size/16), num_classes)

	def forward(self, x):
		# Here you have to define the forward pass
		# Make sure you take care of the skip connections

		if hasattr(self, 'conv1'):
		          x=self.conv1(x)
        		  x=self.bn1(x)
        		  x=self.relu1(x)
        		  x=self.maxpool(x)
		residual=x
		if hasattr(self, 'lyr1conv1'):
		          x=self.lyr1conv1(x)
		          x=self.lyr1bn1(x)
		          x=self.lyr1relu1(x)
		if hasattr(self, 'lyr1conv2'):
		          x=self.lyr1conv2(x)
		          x=self.lyr1bn2(x)
		          x+=residual
		          x=self.lyr1relu2(x)
		residual2=x
		if hasattr(self, 'lyr2conv1'):
		          x=self.lyr2conv1(x)
		          x=self.lyr2bn1(x)
		          x=self.lyr2relu1(x)
		if hasattr(self, 'lyr2conv2'):
		          x=self.lyr2conv2(x)
		          x=self.lyr2bn2(x)
		          x+=residual2
		          x=self.lyr2relu2(x)

		if hasattr(self, 'lyr2conv3'):
		          x=self.lyr2conv3(x)
		          x=self.lyr2bn3(x)
		          x=self.lyr2relu3(x)

		x = x.view(-1, int(input_size*input_size/16))
		#print("final", x.size())
		x=self.fc(x)

		return x

def runLayerRemoval():
    teacher_model = CustomResnet(num_classes=10)
    print("OK")
    teacher_model = torch.load("teacher_net.pt")
    teacher_model.cuda()

    l = LayerRemovalPolicyNetwork()
    features = getLayerInfo(teacher_model)

    student_model = pruneLayers(teacher_model)
    print("OK")

    student_model = train(teacher_model, student_model)
    test(student_model)

runLayerRemoval()
'''

'''

def countParam(model):
	features=getLayerInfo(model)
	parm=len(features)
	return parm

def compressionReward(modelT,modelS):
	paramT = countParam(modelT)
	paramS = countParam(modelS)
	c=paramS/paramT
	c=1-c
	Rc = c*(2-c)
	return Rc

def accuracyReward(model,accuTeacher):
	k=a/accuTeacher
	Ra = min(k,1)
	return Ra


def countParam(model):
	features=getLayerInfo(model)	
	parm=len(features)
	return parm

def compressionReward(modelT,modelS):
	paramT = countParam(modelT)
	paramS = countParam(modelS)
	c=paramS/paramT
	c=1-c
	Rc=c*(2-c)
	return Rc

def accuracyReward(model,accuTeacher)
	#train(model) for 5 epoch
	#calculate acuuracy as a
	k=a/accuTeacher
	Ra=min(k,1)
	return ra


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
# output, (h0, c0) = rnn(input, (h0, c0))
# output, (h0, c0) = rnn(input, (h0, c0))
# output, (h0, c0) = rnn(input, (h0, c0))

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
