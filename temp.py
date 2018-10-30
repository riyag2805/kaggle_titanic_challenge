import pandas as pd
import numpy as np
from scipy import stats

titanic_train=pd.read_csv('C:/Users/riyag/Downloads/all/train.csv')
train_y=titanic_train.iloc[:,1]
x_names=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_X=titanic_train[x_names]
train_X[train_X.isnull().any(axis=1)]
mean_age=np.average(train_X[-train_X.isnull().any(axis=1)].Age)
train_X = train_X.fillna({"Age": mean_age}) #mean of age=29.6
labels_Sex=train_X['Sex'].astype('category').cat.categories.tolist()
replace_Sex={'Sex':{k:v for k,v in zip(labels_Sex,list(range(1,len(labels_Sex)+1)))}}
train_X.replace(replace_Sex,inplace=True)
labels_Embarked=train_X['Embarked'].astype('category').cat.categories.tolist()
replace_Embarked={'Embarked':{k:v for k,v in zip(labels_Embarked,list(range(1,len(labels_Embarked)+1)))}}
train_X.replace(replace_Embarked,inplace=True)
mode_embarked=stats.mode(train_X[-train_X.isnull().any(axis=1)].Embarked)
train_X = train_X.fillna({"Embarked": mode_embarked[0][0]}) #mode of embarked=3

def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=50000 #Setting training iterations
lr=.1 #Setting learning rate
inputlayer_neurons = train_X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

train_y=np.array(train_y)

for i in range(epoch):

#Forward Propogation
    hidden_layer_input1=np.dot(train_X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

#Backpropagation
    E = (train_y-output.T).T
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout = wout+hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += train_X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    
E.T.dot(E)
