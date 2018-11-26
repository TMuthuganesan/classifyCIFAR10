from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
import numpy as np

USE_COLAB         = True

if (USE_COLAB):
  from google.colab import drive
  drive.mount('/content/gdrive')
  DATADIR         = '/content/gdrive/My Drive/datasets/cifar-10-batches-py/'
else:
  DATADIR         = '../../datasets/CIFAR/cifar-10-batches-py'

#global variables
global BATCHSIZE
global LR
global MOMENTUM
BATCHSIZE         = 200
LR                = 0.01
MOMENTUM          = 0.9
EPOCHS            = 20
IPIMG_H           = 32
IPIMG_W           = 32
CHANNELS          = 3
NUM_CLASSES       = 10

#Dataset related
global  lblNames
global  XTrain, yTrain
global  yTrainPredicted   #Labels of training set predicted during feed forward
global  XTest, yTest, XTestGry

def unpickle(file):
  '''
  Reads the CIFAR dataset, returns a dictionary with keys
  'batch_label', 'labels', 'data', 'filenames' 
  '''
  import pickle
  with open(file, 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
  return data

def readLabelNames(dataDir):
  '''
  Read the lable names corresponding to label numbers in dataset, by
  read the metadata file available in dataset.
  '''
  global lblNames
  metaPath    = os.path.join(dataDir,'batches.meta')
  metaData    = unpickle(metaPath)
  lblNames    = metaData[b'label_names']

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
      XTrain    = dataset[dataKey]
      yTrain    = dataset[lblKey]
    else:
      XTrain    = np.concatenate((XTrain,dataset[dataKey]))
      yTrain    = np.concatenate((yTrain,dataset[lblKey]))

def rdTestData(dataDir):
  '''
  Read the test dataset of 10000 images in a loop from dataset directory.
  '''
  global  XTest, yTest
  
  dsPath        = os.path.join(dataDir,'test_batch')
  dataset       = unpickle(dsPath)
  dataKey       = 'data'.encode()
  lblKey        = 'labels'.encode()
  XTest         = dataset[dataKey]
  yTest         = dataset[lblKey]

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

if __name__ == '__main__':
  readLabelNames(DATADIR)
  rdTrainData(DATADIR)
  rdTestData(DATADIR)
  minmaxNormalizeData(train=True)
  zscore(train=True,mean=0.5,stdDev=0.5)
  minmaxNormalizeData(train=False)
  zscore(train=False,mean=0.5,stdDev=0.5)
  
  #Reshape Train and Test data into image of HxWxC
  XTrain    = XTrain.reshape(-1,IPIMG_H,IPIMG_W,CHANNELS)
  XTest     = XTest.reshape(-1,IPIMG_H,IPIMG_W,CHANNELS)
  yTrain    = to_categorical(yTrain, NUM_CLASSES)
  yTest     = to_categorical(yTest, NUM_CLASSES)
  #Construct a model
  model     = Sequential()
  model.add(Conv2D(filters=256, kernel_size=(5, 5), data_format='channels_last',input_shape=(IPIMG_H,IPIMG_W,CHANNELS),activation='relu'))
  model.add(Conv2D(filters=2128,kernel_size=(3, 3),activation='relu'))
  model.add(Dropout(0.5))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(NUM_CLASSES, activation='softmax'))
  
  #optimizer = SGD(lr=LR, momentum=MOMENTUM)
  model.compile(loss='binary_crossentropy',
              optimizer='adam', #optimizer=optimizer
              metrics=['accuracy'])
  model.fit(
        XTrain, yTrain,
        batch_size = BATCHSIZE,
        epochs=EPOCHS,
        validation_data=(XTest, yTest))
        #steps_per_epoch=10,
        #validation_steps=100)
  
  score = model.evaluate(XTest, yTest, batch_size=BATCHSIZE)  #steps=50)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])
