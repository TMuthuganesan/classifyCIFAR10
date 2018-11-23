import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Constants
#Directory where the dataset is located
USE_GRAYIMG       = True
DATADIR           = './datasets/CIFAR/cifar-10-batches-py'
NUM_LAYERS        = 3
NUM_NEURONS_LYR1  = 32
NUM_CLASSES       = 10    #Num of classes in classification problem - its 10 in CIFAR10
EPSILON_INIT      = 0.12
LR                = 0.1
EPOCHS            = 10    #Num of times the learning algo will work through the entire training dataset
BATCHSIZE         = 200   #Num of samples to work through before updating the internal model parameters

#global variables
#Dataset related
global  lblNames
global  XTrain, yTrain
global  yTrainPredicted   #Labels of training set predicted during feed forward
global  XTest, yTest, XTestGry

#Network related
#neuron = [{'a':0, 'w':0, 'z':0, 'd':0}]*NUM_LAYERS
#neuron is a list of dictionaries. Each element in the list corresponds to one layer.
#Each element in the list is a dictionary that comprises of the weights (w), inputs(a),
#internal value z before activations (z), delta from back prop (d)
global  neuron

def unpickle(file):
  '''
  Reads the CIFAR dataset, returns a dictionary with keys
  'batch_label', 'labels', 'data', 'filenames' 
  '''
  import pickle
  with open(file, 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
  return data

def showCIFAR10Img (strIdx,endIdx,dataSet,lblSet):
  '''
  Gets the index, data set & label set, shows the image with label as image name
  '''
  global lblNames
  for i in range (strIdx,endIdx):
    if (USE_GRAYIMG):
      img     = dataSet[i].reshape(32,32)
    else:   
      img     = dataSet[i].reshape(3,32,32).transpose(1,2,0)
      img     = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      
    lblId     = lblSet[i]
    lbl       = lblNames[lblId].decode()
    lbl       = str(i) + '.' + lbl
    print (lbl)
    cv2.imshow(lbl,img)
    
  cv2.waitKey()
  cv2.destroyAllWindows()

def rdTrainData(dataDir):
  '''
  Read all 5 training datasets of 50000 images in a loop from dataset directory.
  '''
  global  XTrain, yTrain
  
  for i in range (1,6): #5 training dataset files
    fName       = 'data_batch_'+str(i)
    dsPath      = os.path.join(dataDir,fName)
    dataset     = unpickle(dsPath)
    dataKey     = 'data'.encode()
    lblKey      = 'labels'.encode()
    if (i==1):
      tempX     = dataset[dataKey]
      yTrain    = dataset[lblKey]
    else:
      tempX     = np.concatenate((tempX,dataset[dataKey]))
      yTrain    = np.concatenate((yTrain,dataset[lblKey]))
    
  if (USE_GRAYIMG):
    m           = tempX.shape[0]
    tempX       = tempX.reshape(m,3,32,32).transpose(0,2,3,1)
    XTrain      = np.zeros((m,32*32),dtype=np.uint8)
    for i in range (m):
      XTrain[i] = cv2.cvtColor(tempX[i], cv2.COLOR_RGB2GRAY).reshape(32*32)
  else:
    XTrain      = tempX

def rdTestData(dataDir):
  '''
  Read the test dataset of 10000 images in a loop from dataset directory.
  '''
  global  XTest, yTest
  
  dsPath        = os.path.join(dataDir,'test_batch')
  dataset       = unpickle(dsPath)
  dataKey       = 'data'.encode()
  lblKey        = 'labels'.encode()
  tempX         = dataset[dataKey]
  yTest         = dataset[lblKey]
  
  if (USE_GRAYIMG):
    m           = tempX.shape[0]
    tempX       = tempX.reshape(m,3,32,32).transpose(0,2,3,1)
    XTest       = np.zeros((m,32*32),dtype=np.uint8)
    for i in range (m):
      XTest[i]  = cv2.cvtColor(tempX[i], cv2.COLOR_RGB2GRAY).reshape(32*32)
  else:
    XTest       = tempX

def readLabelNames(dataDir):
  '''
  Read the lable names corresponding to label numbers in dataset, by
  read the metadata file available in dataset.
  '''
  global lblNames
  metaPath    = os.path.join(dataDir,'batches.meta')
  metaData    = unpickle(metaPath)
  lblNames    = metaData[b'label_names']

def minmaxNormalizeData(train):
  '''Normalize the Train if input arg is true, else normalize the test data.
  Uses Min Max Normalization x-min/(max-min)
  '''
  global XTrain, XTest
  if (train):
    XTrain = (XTrain-np.min(XTrain))/(np.max(XTrain)-np.min(XTrain))
  else:
    XTest = (XTest-np.min(XTest))/(np.max(XTest)-np.min(XTest))

def zscore (train, mean=0.5, stdDev=0.5):
  '''Normalize the Train if input arg is true, else normalize the test data.
  Uses Z Score Normalization x-mean/std. dev. Default mean and std dev = 0.5
  '''
  global XTrain, XTest
  if (train):
    XTrain = (XTrain-mean)/stdDev
  else:
    XTest = (XTest-mean)/stdDev

def quantifyResults (yPredicted):
  '''
  This function quantifies the results and identifies the percentage of 
  correct results predicted against the actual classification in the  test
  dataset.
  '''
  global  yTest
  passCnt     = np.sum(np.array(yPredicted)==np.array(yTest[0:TST_BATCHSIZE]))
  passPrcnt   = (passCnt/len(yPredicted))*100
  return  passPrcnt

def formNeuralNetwork():
  global neuron
  
  #Form neurons and initialize weights
  neuron[0]['a']      = np.zeros(shape=XTrain.shape[1],dtype=np.float64)    #Num of pixels in each data = num of neurons in input layer
  neuron[1]['a']      = np.zeros(shape=NUM_NEURONS_LYR1,dtype=np.float64) #Num of neurons in hidden layer or layer1
  neuron[2]['a']      = np.zeros(shape=NUM_CLASSES,dtype=np.float64)        #Num of output classes required in classification
  
  #Calculation for size of weight matrix
  #If W0 is the weight matrix from layer 0 to layer 1 of a fully connected layer,
  #if there are M neurons in layer0 and N neurons in layer1, then W0 will be of 
  #size N x (M+1). Here there are 1024 neurons in layer0 and 32 neurons in layer1,
  #Hence W0 will be of size 32x1025. +1 is for bias neuron.
  neuron[0]['w']      = np.random.randint(low=0,high=255,size=(NUM_NEURONS_LYR1,XTrain.shape[1]+1),dtype=np.int32)
  neuron[1]['w']      = np.random.randint(low=0,high=255,size=(NUM_CLASSES,NUM_NEURONS_LYR1+1),dtype=np.int32)
  neuron[0]['w']      = (neuron[0]['w']-np.min(neuron[0]['w']))/(np.max((neuron[0]['w']))-np.min((neuron[0]['w'])))
  neuron[1]['w']      = (neuron[1]['w']-np.min(neuron[1]['w']))/(np.max((neuron[1]['w']))-np.min((neuron[1]['w'])))
  neuron[0]['w']      = ((neuron[0]['w']-0.5)/0.5).astype(np.float64)
  neuron[1]['w']      = ((neuron[1]['w']-0.5)/0.5).astype(np.float64)
  #neuron[0]['w'][:,0] = 0 #initialize bias to 0
  #neuron[1]['w'][:,0] = 0 #initialize bias to 0
  #print (neuron[0]['w'])
  #print (neuron[1]['w'])
  #print (neuron[0]['w'].dtype)
  #print (neuron[1]['w'].dtype)

def printTopology():
  print ('\n#############################################################################')
  print ('Number of layers in the network: '+str(len(neuron)))
  print ('Number of nodes in input layer - neuron[0][\'a\'] : '+str(neuron[0]['a'].shape))  #1024
  print ('Number of nodes in hidden layer - neuron[1][\'a\']: '+str(neuron[1]['a'].shape))  #32
  print ('Number of nodes in output layer - neuron[2][\'a\']: '+str(neuron[2]['a'].shape))  #10
  print ('Number of weights from input layer to hidden layer - neuron[0][\'w\']: '+str(neuron[0]['w'].shape)) #32x1025
  print ('Number of weights from hidden layer to output layer - neuron[1][\'w\']: '+str(neuron[1]['w'].shape))  #10x33
  print ('################################################################################\n')
  
def transfer (z,func='sig'):
  '''
  Tranfers the input through a transfer function. For now, the transfer function
  is only sigmoid. Later can be extended to have others.
  '''
  if (func == 'sig'):
    return 1.0 / (1.0 + np.exp(-z))
  
def feedForwardFromLayer(lyr):
  '''
  Feed forward the data from the layer passed as input to the next layer
  '''
  global  neuron
  
  #sigmoid(input*weights)
  #ipWitBias              = np.hstack((1,neuron[lyr]['a'])).astype(np.int32)  #append 1 as the first elt - bias unit
  ipWitBias             = np.hstack((1.0,neuron[lyr]['a'])) #append 1 as the first elt - bias unit
  #print(neuron[lyr]['a'])
  #print(neuron[lyr]['a'].dtype)
  #print (ipWitBias.dtype)
  neuron[lyr+1]['z']    = np.dot(ipWitBias,neuron[lyr]['w'].T)
  #print (neuron[lyr+1]['z'])
  #print (neuron[lyr+1]['z'].dtype)
  neuron[lyr+1]['a']    = transfer(neuron[lyr+1]['z'])
  #print (neuron[lyr+1]['a'])
  #print (neuron[lyr+1]['a'].dtype)
  #quit()

def feedForward (dataIdx):
  '''
  Feed forward for the whole network
  '''
  global  neuron, XTrain, yTrainPredicted
  
  neuron[0]['a']    = XTrain[dataIdx]   #input for the 1st layer is the input for the whole network
  feedForwardFromLayer(0)
  feedForwardFromLayer(1)
  yTrainPredicted[dataIdx]  = np.argmax(neuron[NUM_LAYERS-1]['a'])

def transferDerivative (x,func='sig'):
  '''
  This function calculates derivative of the transer function.
  When the activation function changes, the derivative 
  also changes. By default it calculates the derivative of
  sigmoid function.
  '''
  return x * (1.0 - x)

def calcPredictionError (dataIdx):
  '''This function calculates the error at the output layer.
  At the output layer, the error is the difference
  between actual classification against the training set classification.
  '''
  global neuron, yTrain
  #yTrain is a number indicating the class 0 to 10. This is coded as a list with 1 at
  #the matching class and other indices 0. For example, if yTrain for the data is 6. It
  #indicates that the matching class is 6 for the dataset. This is coded as 
  #[0,0,0,0,0,0,1,0,0,0].
  lyr                             = NUM_LAYERS-1
  yTrainCoded                     = np.zeros(10,dtype=np.float64)
  yTrainCoded[yTrain[dataIdx]]    = 1.0
  #print (yTrainCoded)
  #print(neuron[lyr]['a'])
  err                             = yTrainCoded-neuron[lyr]['a']
  #print (err)
  #print (err.dtype)
  neuron[lyr]['d']                = err * transferDerivative(neuron[lyr]['a'])
  #print (neuron[lyr]['d'])
  #print (neuron[lyr]['d'].dtype)
  
  #dbg prints
  #errDbg = np.sqrt(np.sum(np.square(err)))
  #if (yTrain[dataIdx] == yTrainPredicted[dataIdx]):
   #msg = 'MATCH :: Exp. Lbl = '+str(yTrain[dataIdx])+', Act. Lbl = '+str(yTrainPredicted[dataIdx])+' Err = '+str(errDbg)
  # print (yTrainPredicted[dataIdx])
  #else:
   #msg = 'MISMATCH :: Exp. Lbl = '+str(yTrain[dataIdx])+', Act. Lbl = '+str(yTrainPredicted[dataIdx])+' Err = '+str(errDbg)
  #print (msg)
  return  err

def calcError(dataIdx, lyr):
  '''
  Calculates the error of the data index at layer mentioned in the input.
  This error calculation is not relevant to output layer. Output layer
  error is calculated in calcPredictionError function. 
  In hidden layer, the error is back propagated from the next layer.
  '''
  global neuron, yTrain
  err                             = np.dot(neuron[lyr+1]['d'].T,neuron[lyr]['w'])
  ipWitBias                       = np.hstack((1.0,neuron[lyr]['a']))
  neuron[lyr]['d']                = err * transferDerivative(ipWitBias)

'''
This seems to be wrong compared to machinelearningmaster webpage. LR is not being used here.
Also for the output layer there is no use of input data.
'''
def updateWeights(lr, lyr):
  ipWitBias             = np.hstack((1.0,neuron[lyr]['a'])) #append 1 as the first elt - bias unit
  ipWitBias             = ipWitBias.reshape(ipWitBias.shape[0],1)
  if (lyr==NUM_LAYERS-2):
    wghtChange          = lr*(np.dot(neuron[lyr+1]['d'].reshape(neuron[lyr+1]['d'].shape[0],1), ipWitBias.T))
  else:
    errWithoutBias      = neuron[lyr+1]['d'][1:]
    wghtChange          = lr*(np.dot(errWithoutBias.reshape(errWithoutBias.shape[0],1), ipWitBias.T))
  neuron[lyr]['w']      = neuron[lyr]['w'] + wghtChange
  
def backPropagate(lr,dataIdx):
  err = calcPredictionError(dataIdx)
  calcError(dataIdx,lyr=1)
  updateWeights(lr,lyr=1)
  updateWeights(lr,lyr=0)
  return err
    
  
    
if __name__ == '__main__':
  readLabelNames(DATADIR)
  rdTrainData(DATADIR)
  rdTestData(DATADIR)
  minmaxNormalizeData(train=True)
  zscore(train=True,mean=0.5,stdDev=0.5)
  minmaxNormalizeData(train=False)
  zscore(train=False,mean=0.5,stdDev=0.5)
  #showCIFAR10Img(0,5,XTrain,yTrain)
  
  neuron = [dict() for k in range(NUM_LAYERS)]
  
  #Seed the random number generator, so that results are reproducible with a given seed
  np.random.seed(0)
  
  formNeuralNetwork()
  printTopology()

  yTrainPredicted =np.zeros(yTrain.shape,dtype=np.float32)  #Create a ndarray to store predicted label of the same size as training set
  #Train
  for epoch in range (EPOCHS):
    numBatches    = int(XTrain.shape[0]/BATCHSIZE)
    for batchIdx in range (numBatches):
      dsStrIdx    = batchIdx * BATCHSIZE  #Start index on the dataset for this batch
      dsEndIdx    = dsStrIdx + BATCHSIZE  #End index on the dataset for this batch
      sumSqErr    = 0                     #Clear error for each batch
      for dataIdx in range (dsStrIdx,dsEndIdx):
        feedForward(dataIdx)
        opLyrErr  = backPropagate(LR, dataIdx)
        sumSqErr  = sumSqErr + opLyrErr**2
        #print('neuron[2][\'a\']',neuron[2]['a'])
        #print('neuron[2][\'d\']',neuron[2]['d'])
        #print('neuron[1][\'w\'][:,0]',neuron[1]['w'][:,0])
        #if (dataIdx == 199):
        #  quit()
      MSE         = np.sqrt(sumSqErr)/BATCHSIZE
      #print ('BATCH MSE = ',MSE)
      mismatchCnt = np.sum(yTrainPredicted[dsStrIdx:dsEndIdx] != yTrain[dsStrIdx:dsEndIdx])
      batchErr    = (mismatchCnt/BATCHSIZE)*100
      print('EPOCH: %0d, BATCH: %0d, BATCH FAIL PRCNT: %f'%(epoch,batchIdx,batchErr))
      #if (batchIdx == 249 and epoch == 1):
      #  quit()
