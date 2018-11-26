# classifyCIFAR10
This project has different approaches to classify CIFAR10 dataset. More about 
CIFAR10 dataset can be read at https://www.cs.toronto.edu/~kriz/cifar.html

1. Basic starting point for classification is K-Nearest Neighburs approach 
implemented in knn.py. 

2. Simple back propagation neural network without any framework. This code is 
implemented in SimpleNNWtihBackProp.py. 

3. Simple back propagation neural network similar to approach2 but using PyTorchframework. This code is implemented in PytorchNNWithBackProp_Sigmoid_L1Loss.py

4. Back propagation using PyTorch framework which uses ReLU (Rectified Linear
Unit) and cross entropy loss. This code is implemented in PytorchNNWithBackProp_ReLUSoftMax_CrEntropy.py

5. Now that the bug has bitten me, I wanted to go one more step to implement a
simple convnet which could be run on my not so powerful CPU to classify images.
This code is available at PytorchConvNetGray.py

6. As the results of above experiments were not even close the benchmarks, I
decided to increase the size of the network, with which it became difficult
to train in my CPU, and there were difficulties in moving to cloud with 
PyTorch. Hence an experiment with Keras was done. The implementation is 
available at KerasConvNet_CIFAR10.py


Benckmarks:
Some benchmark results with corresponding papers are available at - http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html


Note: Thanks to Vipul (https://github.com/vaibhawvipul) for taking efforts to
improve the ML, DL ecosystem in India, organising courses, meetups and being 
an active member of the eco-system.
