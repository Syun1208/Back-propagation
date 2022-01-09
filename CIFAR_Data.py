#4. Training Neural Network with Doping Function using CIFAR10 module(3 hidden layers)
#+Layer 1: ReLU function
#+Layer 2: Tanh function 
#+Layer 3: Sigmoid function


import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10
#load datashet
print("Load MNIST Database")
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train=np.reshape(x_train,(50000,3072))/255.0
x_test= np.reshape(x_test,(10000,3072))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def ReLU_derivative(x):
  x[x>0]=1
  x[x<=0]=0
  return x
def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))
def Forwardpass(X,Wh,bh,Wo,bo,Wk,bk,Wl,bl):
    zh = X@Wh.T + bh
    a = np.maximum(zh,0)
    zk=a@Wk.T + bk
    b = np.tanh(zk)
    zl=b@Wl.T + bl
    c = sigmoid(zl)
    zo = c@Wo.T + bo
    o = softmax(zo)
    return o
def AccTest(label,prediction): # calculate the matching score
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy
learningRate = 0.0001
Epoch=50
NumTrainSamples=50000
NumTestSamples=10000
NumInputs=3072
NumHiddenUnits=512
NumClasses=10
#inital weights
#hidden layer 1
Wh=np.matrix(np.random.uniform(-0.5,0.5,(512,3072)))
bh= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh= np.zeros((NumHiddenUnits,NumInputs))
dbh= np.zeros((1,NumHiddenUnits))
#hidden layer 2
Wk=np.matrix(np.random.uniform(-0.5,0.5,(512,512)))
bk= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWk= np.zeros((NumHiddenUnits,NumHiddenUnits))
dbk= np.zeros((1,NumHiddenUnits))
#hidden layer 3
Wl=np.matrix(np.random.uniform(-0.5,0.5,(512,512)))
bl= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWl= np.zeros((NumHiddenUnits,NumHiddenUnits))
dbl= np.zeros((1,NumHiddenUnits))
#Output layer
Wo=np.random.uniform(-0.5,0.5,(10,512))
bo= np.random.uniform(0,0.5,(1,NumClasses))
dWo= np.zeros((NumClasses,NumHiddenUnits))
dbo= np.zeros((1,NumClasses))
from IPython.display import clear_output
loss = []
Acc = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range (Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range (0,NumTrainSamples,Batch_size): 
        #feed fordware propagation
        Batch_samples = Stochastic_samples[ite:ite+Batch_size]
        x = x_train[Batch_samples,:]
        y=y_train[Batch_samples,:]
        zh = x@Wh.T + bh
        a = np.maximum(zh,0)
        zk=a@Wk.T + bk
        b = np.tanh(zk)
        zl=b@Wl.T + bl
        c = sigmoid(zl)
        zo = c@Wo.T + bo
        o = softmax(zo)
        #calculate cross entropy loss
        loss.append(-np.sum(np.multiply(y,np.log10(o))))
        #calculate back propagation error
        do = o-y
        dl =do@Wo
        dk = dl@Wl
        dh = dk@Wk
        #update weight
        dWo = np.matmul(np.transpose(do),c)
        dls = np.multiply(np.multiply(dl,c),1-c)
        dWl = np.matmul(np.transpose(dls),b)
        dks = np.multiply(dk,1-(np.power(b,2)))
        dWk = np.matmul(np.transpose(dks),a)
        dhs = np.multiply(dh,ReLU_derivative(zh))
        dWh = np.matmul(np.transpose(dhs),x)
        dbo = np.mean(do)
        dbl = np.mean(dls)
        dbk = np.mean(dks)
        dbh = np.mean(dhs)
        bo =bo - learningRate*dbo
        bl =bl - learningRate*dbl
        bk =bk-learningRate*dbk
        bh =bh-learningRate*dbh
        Wo =Wo - learningRate*dWo/Batch_size
        Wk =Wk - learningRate*dWk/Batch_size
        Wh =Wh-learningRate*dWh/Batch_size
    #Test accuracy with random innitial weights
    prediction = Forwardpass(x_test,Wh,bh,Wo,bo,Wk,bk,Wl,bl)
    Acc.append(AccTest(y_test,prediction))
    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)],Acc,"o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy test")
    plt.title("Training Neural Network with Doping Function")
    plt.show()
    print('Epoch:', ep )
    print('Accuracy:',AccTest(y_test,prediction))