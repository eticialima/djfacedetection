import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data = pickle.load(open('data_face_features.pickle',mode='rb'))

X = np.array(data['data']) # indendepent variable
y = np.array(data['label']) # dependent variable

X.shape , y.shape

X = X.reshape(-1,128)
X.shape
 
 
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

 





