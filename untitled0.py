from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy
from pandas import read_csv
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
import xgboost
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def buildClassifier():
    classifier=Sequential()
    classifier.add(Dense(units=3,kernel_initializer='uniform',activation='relu' 
                         ,input_dim=7))
    classifier.add(Dropout(p=0.2))
    classifier.add(Dense(units=3,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(p=0.2))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#    sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=False)
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
#classifier=KerasClassifier(build_fn=buildClassifier,batch_size=1,epochs=10)
#output=classifier(train)    
classifier.fit(train_X,train_y,epochs=150,batch_size=25)

predictions = classifier.predict(train_X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
import pandas as pd
import csv
with open('titanic_test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(rounded)

csvFile.close()