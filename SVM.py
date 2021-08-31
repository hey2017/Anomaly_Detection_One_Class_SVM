import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

os.chdir(Directory)

df2 = pd.read_csv('Housing_Data.csv')

data = df2.values

x, y = data[:,:-1], data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)


nu_list = list(np.linspace(0.00001,0.99999,num=100))
#%%
result_lst =[]

for i in nu_list:
    print(i)
    x_train2 = 0
    y_train2 = 0
    y_pred_linear = 0
    y_pred2_linear= 0
    
    model = OneClassSVM(nu = i)
    y_pred = model.fit_predict(x_train)
    tag = y_pred != -1
    x_train2, y_train2 = x_train[tag,:], y_train[tag]
    
    linear_model = LinearRegression()
    #before detecting the outliers
    linear_model.fit(x_train, y_train)
    y_pred_linear = linear_model.predict(x_test)
    #after detecting the outliers
    linear_model.fit(x_train2, y_train2)
    y_pred2_linear = linear_model.predict(x_test)
    
    result_lst.append((i, mean_absolute_error(y_test,y_pred_linear), mean_absolute_error(y_test,y_pred2_linear)))
#%%
result = pd.DataFrame(result_lst)
result.columns = ['nu', 'mae before fit', 'mae after fit']
result = result.sort_values(by='mae after fit', ascending = True)

print('The nu is:\n ', result.iloc[0])

#%%
mae_after = []
contamination_val = []

for i in range(len(result_lst)):
    mae_after.append(result_lst[i][2])
    nu.append(result_lst[i][0])
    
plt.plot(nu, mae_after,'--*')
plt.ylabel('absolute mean square after fit')
plt.xlabel('nu value')
