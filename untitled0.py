#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:47:58 2019

@author: anish
"""

#Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset
movies = pd.read_csv('ml-1m/ml-1m/movies.dat', sep = '::', header = None, 
                     engine = 'python', encoding = 'latin-1')

users = pd.read_csv('ml-1m/ml-1m/users.dat', sep = '::', header = None, 
                     engine = 'python', encoding = 'latin-1')

ratings = pd.read_csv('ml-1m/ml-1m/ratings.dat', sep = '::', header = None, 
                     engine = 'python', encoding = 'latin-1')

#Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

#Converting the data into an array with users in rows and movies in columns
def convert(data):
    new_data = []
    
    for id_users in range(1, nb_users+1):
        #Getting all the rated movies and ratings by user id_users
        id_movies = data[:, 1][data[:, 0] == id_users] 
        id_ratings = data[:, 2][data[:, 0] == id_users]
        """Creating a list representing the ratings of all movies and 
            initializing it to 0"""
        ratings = np.zeros(nb_movies)
        #Populating only those indexes that are rated by the user, rest is 0
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Creating the architecture of the Neural Network
class SAE(nn.Module):
    
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#Training the SAE
nb_epoch = 200 

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.          #No of users who have rated at least 1 movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0: 
            """target.data > 0 returns indexes 
                of the elements of target that is greater than 0"""
            output = sae.forward(input)
            target.require_grad = False
            output[target == 0] = 0 #very important step
            #movies with no rating are again initialized to 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) +
                                             1e-10)
            """When loss.backward() is called, gradients are being computed for
                the weights of the network. These computed gradients are used 
                later on by the optimizer to update the weights."""
            loss.backward()
            optimizer.step()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
    print('epoch '+str(epoch)+' loss: '+str(train_loss/s))


#Testing the SAE
test_loss = 0
s = 0.          #No of users who have rated at least 1 movie
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0: 
        """target.data > 0 returns indexes 
            of the elements of target that is greater than 0"""
        output = sae.forward(input)
        target.require_grad = False
        output[target == 0] = 0 #very important step
        #movies with no rating are again initialized to 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) +
                                         1e-10)
        """When loss.backward() is called, gradients are being computed for
            the weights of the network. These computed gradients are used 
            later on by the optimizer to update the weights."""
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('Test loss: '+str(test_loss/s))














