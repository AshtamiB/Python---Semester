# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 07:07:31 2018

@author: Ashtami
Student Number: 18201912
Subject: STAT40800 Data Programming with Python 
Date: 18/12/2018
"""

import os
os.chdir("C:/Users/Ashtami/Documents/Python/")


#####Q2######

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

#----------- Pre-Procesing of Timeseries with Pandas --------------#

#### Q2 (a) #####
data_frame = pd.read_csv('TiSeries.csv', header=0)

##### Q2 (b) ####
data_frame['Date'] = pd.to_datetime(data_frame['Date'])
indexed_df = data_frame.set_index('Date')
timeseries = indexed_df['Value']


#--------------------- Some Timeseries Analysis -------------------#
##### Q2 (c) #####

lag_acf = acf(timeseries, nlags=20)
lag_pacf = pacf(timeseries, nlags=20)
rolmean = pd.rolling_mean(timeseries, window=49)
rolstd = pd.rolling_std(timeseries, window=49)


#### Q2 (d) #######
size = int(len(timeseries) - 17)
train, test = timeseries[0:size], timeseries[size:len(timeseries)]

history = [x for x in train]


##### Q2 (e) #######
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
xval_err = 0
for t in range(len(test)):
    model = ARIMA(history, order=(2,0,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    e = yhat - obs
    xval_err += np.dot(e,e)
    rmse = np.sqrt(xval_err/len(test))
    print('predicted=%f, expected=%f' % ((yhat), (obs)))
    
    
##### Q2 (f) ######
print(predictions)

plt.subplot(221)
plt.plot(timeseries, color='black', label='Original Timeseries', marker='o')
plt.plot(predictions, color='red', label='Predicted Values', marker='v')
plt.legend(loc='best')
plt.title('Time Series, Predicted Values')
plt.show()

##Alternative to Q2 (f) ######
plt.subplot(221)
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('TiSeries.csv', header=0)
series.plot(style='k.')
plt.legend(loc='best')
plt.title('Time Series')


plt.subplot(222)
plt.plot(predictions, color='red', label='Predicted Values', marker='v')
plt.legend(loc='best')
plt.title('Predicted Values')
plt.show()
pyplot.show()


###### Q2 (g) ########
#RMSE value is being computed in the For loop of Q2 (e)
print('The Root Mean Square Error (RMSE) of predicted and expected value of test data is:%f' % rmse)