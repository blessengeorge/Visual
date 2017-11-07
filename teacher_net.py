from __future__ import division, print_function, unicode_literals

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn

#get_ipython().magic(u'matplotlib inline')
#import matplotlib.pyplot as plt
from torch.autograd import Variable

from PIL import Image

batch_size = 100
num_epochs = 1
learning_rate = 0.01


#	Creates a dictionary for class name to index/label conversion
def class_to_index(root):
	class_list = sorted([directory for directory in os.listdir(root)])
	class_to_labels = {class_list[i]: i for i in range(len(class_list))}
	return class_to_labels

# 	Creates a list of image file path and label pairs
def create_dataset(root, class_to_labels):
	dataset = []
	for label in sorted(class_to_labels.keys()):
		path = os.path.join(root, label)
		for image_file in os.listdir(path):
			image_file = os.path.join(path, image_file)
			if os.path.isfile(image_file):
				dataset.append((image_file, class_to_labels[label]))
	return dataset


class CDATA(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, train, transform=None):
	self.root = root_dir
	self.train = train
	self.transform = transform

	if train:
		image_folder = os.path.join(self.root, "train")

	else:
		image_folder = os.path.join(self.root, "test")

	class_to_labels = class_to_index(image_folder)
	self.dataset = create_dataset(image_folder, class_to_labels)

    def __len__(self):
	return len(self.dataset)

    def __getitem__(self, idx):
	image_path, label = self.dataset[idx]
	image = Image.open(image_path).convert('RGB')
	if self.transform is not None:
		image = self.transform(image)
	return (image, label)

input_size = 32
composed_transform = transforms.Compose([transforms.Scale((input_size,input_size)),transforms.ToTensor()])
train_dataset = CDATA(root_dir='./CDATA/notMNIST_small', train=True, transform=composed_transform)
test_dataset = CDATA(root_dir='./CDATA/notMNIST_small', train=False, transform=composed_transform)

print('Size of train dataset: %d' % len(train_dataset))
print('Size of test dataset: %d' % len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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

		#print("first", x.size())
		x=self.conv1(x)
		#print("conv1", x.size())
		x=self.bn1(x)
		x=self.relu1(x)
		x=self.maxpool(x)
		#print("maxpool", x.size())
		residual=x
		x=self.lyr1conv1(x)
		#print("lyr1conv1", x.size())
		x=self.lyr1bn1(x)
		x=self.lyr1relu1(x)
		x=self.lyr1conv2(x)
		#print("lyr1conv2", x.size())
		x=self.lyr1bn2(x)
		x+=residual
		x=self.lyr1relu2(x)
		residual2=x
		x=self.lyr2conv1(x)
		#print("lyr2conv1", x.size())
		x=self.lyr2bn1(x)
		x=self.lyr2relu1(x)
		x=self.lyr2conv2(x)
		#print("lyr2conv2", x.size())
		x=self.lyr2bn2(x)
		x+=residual2
		x=self.lyr2relu2(x)
		#print(x.size())

		x=self.lyr2conv3(x)
		#print("lyr2conv2", x.size())
		x=self.lyr2bn3(x)
		#x+=residual2
		x=self.lyr2relu3(x)
		#print(x.size())

		x = x.view(-1, int(input_size*input_size/16))
		#print("final", x.size())
		x=self.fc(x)

		return x

model = CustomResnet(num_classes = 10) # 100 classes since CIFAR-100 has 100 classes

# Load CIFAR-100 weights. (Download them from assignment page)
# If network was properly implemented, weights should load without any problems
# model.load_state_dict(torch.load('./cifar')) # Supply the path to the weight file

# Change last layer to output 10 classes since our dataset has 10 classes
model.fc =nn.Linear(model.fc.in_features, 10) # Complete this statement. It is similar to the resnet18 case

model.cuda()

# Loss function and optimizers
criterion = nn.CrossEntropyLoss()# Define cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)# Use Adam optimizer, use learning_rate hyper parameter

def train():
	#NOTE the change from iter to enumerate
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images).cuda()
			labels = Variable(labels).cuda()
			optimizer.zero_grad()
			outputs = model(images)
			error = criterion(outputs, labels)
			error.backward()
			optimizer.step()
			if (i+1) % (1000/batch_size) == 0:
				print (error.data[0])
		print("Epoch Complete!")
			# Make sure to output a matplotlib graph of training losses

def test():
	correct = 0
	total = 0

	for images, labels in test_loader:
		#print(images)
		images = Variable(images)

		#if(use_gpu):
		images = images.cuda()

		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		#print(predicted.cpu())
		correct += (predicted.cpu() == labels.cpu()).sum()
	print(correct, total)
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

train()
test()
torch.save(model, "teacher_net.pt")
