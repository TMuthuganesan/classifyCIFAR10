import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as AG
import torchvision as tv
import numpy as np

USE_GRAYIMG       = True
DATADIR           = 'D:\\general\\ML_DL\\DLCourse_VipulVaibhav\\datasets\\CIFAR'
NUM_CLASSES       = 10 #Num of classes in classification problem - its 10 in CIFAR10
LOG_INTERVAL      = 10
IMG_W             = 32
IMG_H             = 32

#global variables
global BATCHSIZE
global LR
global MOMENTUM
BATCHSIZE         = 200
LR                = 0.01
MOMENTUM          = 0.9
EPOCHS            = 20

#Build conv - Relu - conv - relu - max pooling - fc1 - fc2 - fc3




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
  def __init__(self):
    super(Neuron, self).__init__()
    #Trail 1 Network
    self.conv1      = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,stride=1,padding=0)
    nn.init.xavier_uniform_(self.conv1.weight)
    self.conv2      = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5,stride=1,padding=0)
    nn.init.xavier_uniform_(self.conv2.weight)
    self.conv2_bn   = nn.BatchNorm2d(8)
    self.conv2_drop = nn.Dropout2d()
    self.pool       = nn.MaxPool2d(2, 2)
    self.fc1        = nn.Linear(8 * 12 * 12, 128)
    nn.init.xavier_uniform_(self.fc1.weight)
    self.fc1_bn     = nn.BatchNorm1d(128)
    self.fc2        = nn.Linear(128, 64)
    nn.init.xavier_uniform_(self.fc2.weight)
    self.fc2_bn     = nn.BatchNorm1d(64)
    self.fc3        = nn.Linear(64,10)
    nn.init.xavier_uniform_(self.fc3.weight)
    
    #Trial 2 Network - Reach error of 0.54 and Fail% of 16.5% after 20 epochs with LR=0.1, momentum = 0.9
    #Test dataset generates failure of 46.07%
    #self.conv1      = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,stride=1,padding=0)
    #self.conv2      = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5,stride=1,padding=0)
    #self.pool       = nn.MaxPool2d(2, 2)
    #self.fc1        = nn.Linear(16 * 12 * 12, 128)
    #self.fc2        = nn.Linear(128, 64)
    #self.fc3        = nn.Linear(64,10)

  def forward(self,x):
    x = F.relu(self.conv1(x)) #32x32 image to 32 28x28 images through 16 filter kernels
    x = F.relu(self.conv2_drop(self.conv2_bn(self.conv2(x)))) #32 28x28 images to 16 24x24 images
    x = self.pool(x)          #16 24x24 image to 8 12x12 by downsampling with max value per each 2x2 block
    x = x.view(-1, 8 * 12 * 12)
    x = F.relu(self.fc1_bn(self.fc1(x)))
    x = F.relu(self.fc2_bn(self.fc2(x)))
    x = F.relu(self.fc3(x))
    return F.log_softmax(x)

if __name__ == '__main__':
  trainLoader     = rdDataSet(DATADIR,train=True) 
  testLoader      = rdDataSet(DATADIR,train=False)
  
  ###############Create and configure the network#######################
  neuron = Neuron() 
  print (neuron)

  #Create a stochastic gradient descent optimizer
  optimizer     = optim.SGD(neuron.parameters(), lr=LR, momentum=MOMENTUM)
  scheduler     = scheduler.ReduceLROnPlateau(optimizer, 'min')
  
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
      
      if (batchIdx % LOG_INTERVAL == 0):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch Fail%: {:.4f}'.format(
                    epoch, batchIdx * len(data), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.data[0],batchErr))
  
  print ('Training done!!!')
  
  #######################Test the network###############################
  mismatchCnt = 0
  loss        = 0
  for (data, target) in (testLoader):      
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
