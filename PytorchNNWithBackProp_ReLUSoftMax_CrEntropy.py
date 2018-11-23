import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as AG
import torchvision as tv
import numpy as np

USE_GRAYIMG       = True
DATADIR           = './datasets/CIFAR'
NUM_NEURONS_LYR1  = 32
NUM_CLASSES       = 10 #Num of classes in classification problem - its 10 in CIFAR10
LOG_INTERVAL      = 10
IMG_W             = 32
IMG_H             = 32

#global variables
global BATCHSIZE
global LR
global MOMENTUM
BATCHSIZE         = 200
LR                = 0.1
MOMENTUM          = 0.9
EPOCHS            = 20

def rdDataSet(dataDir,train=True):
  '''
  Using torchvision load training dataset from the given data folder
  '''
  global BATCHSIZE
  transformOps    = [tv.transforms.Grayscale(num_output_channels=1),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
  dataTransform   = tv.transforms.Compose(transformOps)
  dataSet         = tv.datasets.CIFAR10(dataDir, train=train, download=False, transform=dataTransform)
  dataLoader      = data.DataLoader(dataSet, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
  return dataLoader

class Neuron(nn.Module):
  def __init__(self, ipNodes,lyr2Nodes,opNodes):
    super(Neuron, self).__init__()
    self.lyr1Nodes  = ipNodes
    self.lyr2Nodes  = lyr2Nodes
    self.lyr3Nodes  = opNodes
    self.fc1        = nn.Linear(self.lyr1Nodes, self.lyr2Nodes)
    self.fc2        = nn.Linear(self.lyr2Nodes,self.lyr3Nodes)
    
  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return F.log_softmax(x)

if __name__ == '__main__':
  trainLoader     = rdDataSet(DATADIR,train=True) 
  testLoader      = rdDataSet(DATADIR,train=False)
  
  ###############Create and configure the network#######################
  neuron = Neuron(IMG_H*IMG_W,NUM_NEURONS_LYR1,NUM_CLASSES) 
  print (neuron)

  #Create a stochastic gradient descent optimizer
  optimizer    = optim.SGD(neuron.parameters(), lr=LR, momentum=MOMENTUM)
  
  #Create a loss function
  criterion = nn.CrossEntropyLoss()  #Cross entropy loss - combines nn.LogSoftmax() and nn.NLLLoss()
  #criterion    = nn.NLLLoss() #Negative log likelihood loss - gives the cross entropy loss
  
  #######################Train the network##############################
  for epoch in range (EPOCHS):
    for batchIdx, (data, target) in enumerate(trainLoader):
      #batchIdx - Index of the batch within each epoch
      #data - CIFAR image data read by DataLoader in batches of size mentioned in DataLoader call
      #target - Expected label indicating the class of the data
      
      #Convert data and the labels into PyTorch variables which can be used with Autograd
      data        = AG.Variable(data)
      target      = AG.Variable(target)
      
      #Reshape the data from image format to a vector format
      (B,C,H,W)   = data.size() #Batch size, Num of Channels, Height, Width
      data        = data.view(B,H*W)
      
      #Reset all gradients in the model
      optimizer.zero_grad()
      
      #Forward pass the data batch
      nwOut       = neuron(data)  #calls forward defined in Neuron class
      classOut    = torch.argmax(nwOut,1)
      
      #Calculate the failure in percentage by compaing the predicted class with expected class
      mismatchCnt = torch.sum(classOut != target)
      batchErr    = (mismatchCnt.type(torch.float32)/BATCHSIZE)*100
      
      #Calculate the loss
      loss        = criterion(nwOut, target)
      
      #Bckward pass the loss
      loss.backward()
      
      #Gradient descent
      optimizer.step()  #GD based on grandients calculated in above back prop
      
      #if (batchIdx % LOG_INTERVAL == 0):
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch Fail%: {:.4f}'.format(
                    epoch, batchIdx * len(data), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.data[0],batchErr))
  
  print ('Training done!!!')
  
  #######################Test the network###############################
  mismatchCnt = 0
  loss        = 0
  for (data, target) in (testLoader):
    
    #Reshape the data from image format to a vector format
    (B,C,H,W)     = data.size() #Batch size, Num of Channels, Height, Width
    data          = data.view(B,H*W)
      
    #Forward pass the data batch
    nwOut         = neuron(data)  #calls forward defined in Neuron class
    classOut      = torch.argmax(nwOut,1)
    
    #Calculate the failure in percentage by compaing the predicted class with expected class
    mismatchCnt   = mismatchCnt + torch.sum(classOut != target) #accumulate mismatch count for each batch in all test images
    
    #Calculate the loss
    loss          = loss + criterion(nwOut, target)
    
  testErr         = (mismatchCnt.type(torch.float32)/len(testLoader.dataset))*100
  testLoss        = loss / BATCHSIZE
  
  print ('###################################################################################')
  print ('Loss: {:.6f}\tBatch Fail%: {:.4f}'.format(testLoss,testErr))
  print ('###################################################################################')
