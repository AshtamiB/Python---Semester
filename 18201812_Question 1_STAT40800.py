# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 07:07:26 2018

@author: Ashtami
Student Number: 18201912
Subject: STAT40800 Data Programming with Python 
Date: 18/12/2018
"""
####Question 1 (a)#####

import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import sklearn.preprocessing as skp
from statsmodels.tools import categorical
import warnings
import sklearn.model_selection as skms
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
warnings.filterwarnings('ignore')
#----------- -----------Pre Pre-Procesing of Timeseries with Pandas
dframe = pd.read_csv('Q1Data.csv', header=None) #header none because no column names
dframe.info()
numdframe=dframe.iloc[:,1:]
catdframe=dframe.iloc[:,0]
catdf_encod=categorical(catdframe.values,dictnames=False,drop=True)
numArr=np.asarray(numdframe.values)
catArr=np.asarray(catdf_encod)
Output=numArr[:,5]
Inp_num=numArr[:,0:5]
Input = np.concatenate((Inp_num, catArr), axis=1)
Input=np.c_[catArr,Inp_num]
print(Input.shape)


####Q1 (b)########

imp = skp.Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
Input_new=imp.fit_transform(Input)



####Q1 (c)######

X_train,X_test,y_train,y_test=skms.train_test_split(Input_new,Output,test_size=0.25,random_state=111)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(Input)



##### Q1 (d)######

svc_rbf=SVC(kernel='rbf',gamma='auto',probability=False,tol=0.001,max_iter=1,random_state=None)
svc_rbf.fit(X_train,y_train)

svc_rbf_train_pred=svc_rbf.predict(X_train)
svc_rbf_test_pred=svc_rbf.predict(X_test)

svc_poly=SVC(kernel='poly',degree=3,gamma='auto',probability=False,tol=0.001,max_iter=1,random_state=None)
svc_poly.fit(X_train,y_train)

svc_poly_train_pred=svc_poly.predict(X_train)
svc_poly_test_pred=svc_poly.predict(X_test)


###### Q1 (e) #######

#Confusion Matrix for RBF Kernel
CM_rbf = confusion_matrix(y_test,svc_rbf_test_pred)
print(CM_rbf)

#Confusion Matrix for Poly Kernel
CM_poly = confusion_matrix(y_test,svc_poly_test_pred)
print(CM_poly)

print(X_train)

#### Q1 (f) #####

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =X_train[:,1]
y =X_train[:,2]
z =X_train[:,3]
#print(x)
ax.scatter(x, y, z, c='r', marker='o', s=150)
ax.set_xlabel('Gender')
ax.set_ylabel('Age')
ax.set_zlabel('Diastolic')
#ax.scatter(X_train[:,0],X_train[:,1],X_train[:,2],c='black', s=200)

ax.scatter(x,color='red')
ax.scatter(y,color='blue')
ax.scatter(z,color='green')
plt.legend(loc='best')
plt.show()



###### Q1 (g) #######

# Random Forest (RF) with 4 trees
RF=RandomForestClassifier (n_estimators=4,random_state=12)
RF.fit(X_train,y_train)
y_rfpredict=RF.predict(X_test)

# 4- fold cross validation
kf = KFold(len(X_train), n_folds=4)
xval_err = 0
RF=RandomForestClassifier (n_estimators=4,random_state=12)
for train,test in kf:
    RF.fit(X_train,y_train)
    y_rfpredict=RF.predict(X_test)
    e = y_rfpredict - y_test
    xval_err += np.dot(e,e)
    rmse_4cv = np.sqrt(xval_err/len(x))
    
print('Random Forest RMSE on 4-fold CV: %.4f' %rmse_4cv)

