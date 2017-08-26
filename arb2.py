
from __future__ import division, print_function, unicode_literals

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from torch.autograd import Variable 

from PIL import Image



batch_size = 1
num_epochs = 5
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


composed_transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor()])
train_dataset = CDATA(root_dir='/home/george/py_programs/CDATA/notMNIST_small', train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = CDATA(root_dir='/home/george/py_programs/CDATA/notMNIST_small/', train=False, transform=composed_transform) # Supply proper root_dir

# Let's check the size of the datasets, if implemented correctly they should be 16854 and 1870 respectively
print('Size of train dataset: %d' % len(train_dataset))
print('Size of test dataset: %d' % len(test_dataset))


# Create loaders for the dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


vgg16 = models.vgg16(pretrained=True)
#resnet18 = models.resnet18(pretrained=True)

# Code to change the last layers so that they only have 10 classes as output
vgg16.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 10),
)

criterion = nn.CrossEntropyLoss()	# Define cross-entropy loss			
###### WAAAAITTTT!!
	# What is the difference between CrossEntropyLoss() and CrossEntropyLoss???? I got an error because of this saying CrossEntropyLoss object has
	# no attribute backward()

optimizer_vgg16 = torch.optim.Adam(vgg16.parameters(), lr=learning_rate)	# Use Adam optimizer, use learning_rate hyper parameter


def train_vgg16(train_loader, optimizer):

	#for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images)
			labels = Variable(labels)
			optimizer_vgg16.zero_grad()
			print(i)
			outputs = vgg16(images)
			error = criterion(outputs, labels)
			error.backward()
			optimizer_vgg16.step()
			if (i+1) % 100 == 0:
                		print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                       		%(epoch+1, num_epochs, i+1, len(train_data)//batch_size, error.data[0]))
    
def train_resnet18():
    # Same as above except now using the Resnet-18 network
	pass


def test(model):
	pass


