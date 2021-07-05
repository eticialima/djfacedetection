import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

data = pickle.load(open('data_face_features.pickle',mode='rb'))

X = np.array(data['data']) # indendepent variable
y = np.array(data['label']) # dependent variable

X.shape , y.shape

X = X.reshape(-1,128)
X.shape
 
 
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

model_logistic = LogisticRegression()
model_logistic.fit(x_train,y_train) # training logistic regression

def get_report(model, x_train,y_train,x_test,y_test):
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # accuracy score
    acc_train = accuracy_score(y_train,y_pred_train)
    acc_test = accuracy_score(y_test,y_pred_test)

    # f1 score
    f1_score_train = f1_score(y_train,y_pred_train,average='macro')
    f1_score_test = f1_score(y_test,y_pred_test,average='macro')


    print('Accuracy Train = %0.2f'%acc_train)
    print('Accuracy Test = %0.2f'%acc_test)
    print('F1 Score Train = %0.2f'%f1_score_train)
    print('F1 Score Test = %0.2f'%f1_score_test)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
 