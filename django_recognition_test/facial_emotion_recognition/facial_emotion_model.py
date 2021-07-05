import numpy as np
import pandas as pd
import pickle

data = pickle.load(open('data_face_features_emotion.pickle',mode='rb'))

X = np.array(data['data']) # indendepent variable
y = np.array(data['label']) # dependent variable

X.shape , y.shape

X = X.reshape(-1,128)
X.shape

# split the data into train and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

## Train Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

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
    
get_report(model_logistic,x_train,y_train,x_test,y_test)

# Support Vector Machines
model_svc = SVC(probability=True)
model_svc.fit(x_train,y_train) 
get_report(model_svc,x_train,y_train,x_test,y_test)

# Random Forest
model_rf = RandomForestClassifier(n_estimators=10,)
model_rf.fit(x_train,y_train)
get_report(model_rf,x_train,y_train,x_test,y_test)

# Voting Classifier
model_voting = VotingClassifier(estimators=[
    ('logistic',LogisticRegression()),
    ('svm',SVC(probability=True)),
    ('rf',RandomForestClassifier())
], voting='soft',weights=[2,3,1])

model_voting.fit(x_train,y_train)
get_report(model_voting,x_train,y_train,x_test,y_test)

# Parameter Tuning
from sklearn.model_selection import GridSearchCV
model_grid = GridSearchCV(model_voting,
                         param_grid={
                             'svm__C':[3,5,7,10],
                             'svm__gamma':[0.1,0.3,0.5],
                             'rf__n_estimators':[5,10,20],
                             'rf__max_depth':[3,5,7],
                             'voting':['soft','hard']
                         },scoring='accuracy',cv=3,n_jobs=1,verbose=2)

model_best_estimator = model_grid.best_estimator_
model_grid.best_score_
pickle.dump(model_best_estimator,open('./models/machinelearning_face_emotion.pkl',mode='wb'))