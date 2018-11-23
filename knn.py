import os
import cv2
import numpy as np

#Directory where the dataset is located
dataDir				= './datasets/CIFAR/cifar-10-batches-py'

#Configuration constants
TST_BATCHSIZE	= 100					#Size of test images used for classification 
K							= 5
DIST_MTD			= 'Manhattan' #'Euclidean' or 'Manhattan'
USE_GRAYIMG		= False				#Choose gray or 3 ch. color image for classification

#global variables
global	lblNames
global	XTrain, yTrain, XTrainGry
global	XTest, yTest, XTestGry

def unpickle(file):
	'''
	Reads the CIFAR dataset, returns a dictionary with keys
	'batch_label', 'labels', 'data', 'filenames' 
	'''
	import pickle
	with open(file, 'rb') as fo:
		data = pickle.load(fo, encoding='bytes')
	return data

def showCIFAR10Img (idx,dataSet,lblSet):
	'''
	Gets the index, data set & label set, shows the image with label as image name
	'''
	global lblNames
	img			= dataSet[idx].reshape(3,32,32).transpose(1,2,0)
	if (USE_GRAYIMG):
		img			= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		img			= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	lblId		= lblSet[idx]
	lbl			= lblNames[lblId].decode()
	print (lbl)
	cv2.imshow(lbl,img)

def readNShowDataSet (dataDir, dataFileName, metaFileName):
	'''
	This is a simple debug function to read a few sample data from
	dataset and show that using its classified label in metadata as
	figure name.
	'''
	global lblNames
	
	dsPath			= os.path.join(dataDir,dataFileName)
	metaPath		= os.path.join(dataDir,metaFileName)
	
	#read dataset and labels from meta data
	dataset1		= unpickle(dsPath)
	metaData		= unpickle(metaPath)
	lblNames		= metaData[b'label_names']	
	dataKey			= 'data'.encode()
	lblKey			= 'labels'.encode()
	dataSet			= dataset1[dataKey]
	lblSet			= dataset1[lblKey]
	
	for i in range (5):
		showCIFAR10Img(i,dataSet,lblSet)
	
	cv2.waitKey()
	cv2.destroyAllWindows()

def rdTrainData(dataDir):
	'''
	Read all 5 training datasets of 50000 images in a loop from dataset directory.
	'''
	global	XTrain, yTrain, XTrainGry
	
	
	for i in range (1,6):	#5 training dataset files
		fName				= 'data_batch_'+str(i)
		dsPath			= os.path.join(dataDir,fName)
		dataset			= unpickle(dsPath)
		dataKey			= 'data'.encode()
		lblKey			= 'labels'.encode()
		if (i==1):
			XTrain		= dataset[dataKey]
			yTrain		= dataset[lblKey]
		else:
			XTrain		= np.concatenate((XTrain,dataset[dataKey]))
			yTrain		= np.concatenate((yTrain,dataset[lblKey]))
		
	if (USE_GRAYIMG):
		m					= XTrain.shape[0]
		XTrain		= XTrain.reshape(m,3,32,32).transpose(0,2,3,1)
		XTrainGry	= np.zeros((m,32,32))
		for i in range (m):
			XTrainGry[i]		= cv2.cvtColor(XTrain[i], cv2.COLOR_RGB2GRAY)

def rdTestData(dataDir):
	'''
	Read the test dataset of 10000 images in a loop from dataset directory.
	'''
	global	XTest, yTest, XTestGry
	
	dsPath			= os.path.join(dataDir,'test_batch')
	dataset			= unpickle(dsPath)
	dataKey			= 'data'.encode()
	lblKey			= 'labels'.encode()
	XTest				= dataset[dataKey]
	yTest				= dataset[lblKey]
	
	if (USE_GRAYIMG):
		m					= XTest.shape[0]
		XTest			= XTest.reshape(m,3,32,32).transpose(0,2,3,1)
		XTestGry	= np.zeros((m,32,32))
		for i in range (m):
			XTestGry[i]		= cv2.cvtColor(XTest[i], cv2.COLOR_RGB2GRAY)

def readLabelNames(dataDir):
	'''
	Read the lable names corresponding to label numbers in dataset, by
	read the metadata file available in dataset.
	'''
	global lblNames
	metaPath		= os.path.join(dataDir,'batches.meta')
	metaData		= unpickle(metaPath)
	lblNames		= metaData[b'label_names']

def KNNClassify(tstData,k=5,distMtd='Manhattan'):
	'''
	This Function calssifies the incoming test data using K Nearest Neighbours.
	K shall be specified as an input argument by default is 5.
	Distance to measure the proximity is based on the argument distMtd, default
	is simple Sum of Absolute Differences or Manhattan distance. Euclidean
	distance is the other distance measure. A global configuration constant
	configures whethere the algorithm runs on 3-channel color images or gray 
	image generated from the RGB image.
	'''
	global	yTest
	yOut					= []
	numTstData		= tstData.shape[0]
	print ('Number of test samples to classify = %d'%numTstData)
	for i in range (numTstData):
		if (distMtd == 'Manhattan'):
			if (USE_GRAYIMG):
				dist				= np.sum(np.abs(XTrainGry-tstData[i]),axis=1)
			else:
				dist				= np.sum(np.abs(XTrain-tstData[i]),axis=1)
		elif  (distMtd == 'Euclidean'):
			if (USE_GRAYIMG):
				dist				= np.sqrt(np.sum(np.square(XTrainGry-tstData[i]),axis=1))
			else:
				dist				= np.sqrt(np.sum(np.square(XTrain-tstData[i]),axis=1))
		kMinIndices = np.argsort(dist)[0:k]
		idx,cnts		= np.unique(kMinIndices,return_counts=True)
		cmnIdx			= idx[np.argmax(cnts)]	#index that occurs max num of times among the K indices
		yOut.append(yTrain[cmnIdx])
		if (i%100 == 0):
			print ('Sample %d done'%i)
	return yOut

def quantifyResults (yPredicted):
	'''
	This function quantifies the results and identifies the percentage of 
	correct results predicted against the actual classification in the  test
	dataset.
	'''
	global	yTest
	passCnt			= np.sum(np.array(yPredicted)==np.array(yTest[0:TST_BATCHSIZE]))
	passPrcnt		= (passCnt/len(yPredicted))*100
	return	passPrcnt
	
if __name__ == '__main__':
	#readNShowDataSet(dataDir, 'data_batch_1', 'batches.meta')
	readLabelNames(dataDir)
	rdTrainData(dataDir)
	rdTestData(dataDir)
	if (USE_GRAYIMG):
		yPredicted	= KNNClassify(XTestGry[0:TST_BATCHSIZE],K,DIST_MTD)
	else:
		yPredicted	= KNNClassify(XTest[0:TST_BATCHSIZE],K,DIST_MTD)
	passPrcnt		= quantifyResults(yPredicted)
	print ('Accuracy of algorithm is = %d'%passPrcnt)
	
	
	
	
	
	
	
