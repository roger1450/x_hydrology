"""Use Machine Learning (ML) methods to surrogate traditional climate models, and make predictions.
Author = Fa Li

.. MIT License
..
.. Copyright (c) 2019 Fa Li
"""
from netCDF4 import Dataset
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

def data_loading(data_dir,file_variables):#
    """**Description:** this method is used to load the required climate variables

        Parameters
        ----------
        data_dir: str,default=""
            the directory of all files corresponding to required climate variables
        file_names: {}, default={}
            a dictionary mapping each file name to its corresponding variable name
        Return
        ----------
        output: list
            a list containing all variables used for further trainging and test through machine learning
        Examples
        --------
        data_dir='temp/'
        file_names={'file_name1.nc': 'temperature'}
        output=data_loading(data_dir,file_names)#outpiut is a list of the required variables (np.ndarray)
    """
    assert len(file_variables)>0 and type(file_variables)==dict
    output=[]
    for file in file_variables:
        file_path=data_dir+file
        var_name=file_variables[file]
        nc_file=Dataset(file_path, 'r')
        var_value =nc_file[var_name][:]
        nc_file.close()
        output.append(var_value)
    return output
def train(vars,target_idx,time_lag,time_dim,lat_dim,lon_dim,para_path='para_path',save=True):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            vars: list,default=[]
                the list of all required climate variables
            target_idx: list, default=[]
                the list of indexes corresponding to predicted variable
            time_lag: int, default=1
                time delay of climate variables
            time_dim: int
                the corresponding dimension index of 'time' in climate variables
            lat_dim: int
                the corresponding dimension index of 'latitude' in climate variables
            lon_dim: int
                the corresponding dimension index of 'longitude' in climate variables
            para_path: str
                the directory for saving parameters of models during training
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            None: The return value will be designed for next processing
            Examples
            --------
            root_dir = ''
            file_variables = {'Temperature_2.nc': 'temperature'}
            time_lag = 2
            time_dim, lat_dim, lon_dim = 0, 2, 3
            vars = data_loading(root_dir, file_variables)
            output=data_loading(data_dir,file_names)
            target_idx = [0]
            train(vars, target_idx, time_lag, time_dim, lat_dim, lon_dim, para_path)

        """
    if not os.path.exists(para_path):
        os.makedirs(para_path)
    dims = vars[0].shape
    time_length, rows, cols = dims[time_dim] - time_lag, dims[lat_dim], dims[lon_dim]
    total_task_num=rows*cols*len(target_idx)
    task_idx=0
    for lat in range(rows):
        for lon in range(cols):
            input = np.zeros((time_length, len(vars) * time_lag))
            output=np.zeros((time_length, len(target_idx)))
            output_cnt=0
            for var_idx in range(len(vars)):
                if len(vars[var_idx].shape)!=3:
                    vars[var_idx]=np.squeeze(vars[var_idx])
                series=vars[var_idx][:, lat, lon]
                for lag in range(1,time_lag+1):
                    input[:,var_idx*time_lag+(lag-1)]=series[(time_lag-lag):((time_lag-lag)+time_length)]
                if var_idx in target_idx:
                    output[:,output_cnt]=series[time_lag:(time_lag+time_length)]
                    output_cnt+=1
            for target in range(len(target_idx)):
                Y=output[:,target]
                Y_predict=ML_mdoel(X=input,Y=Y,model='decision tree',file_name=str(lat)+'_'+str(lon)+'_'+str(target),save=save)
                print('relative error is:',Relative_Error(Y,Y_predict))
                if (task_idx + 1)%5==0:
                    print("processing:{0}%".format(round((task_idx + 1) * 100 / total_task_num,5)))
                    time.sleep(0.01)
                task_idx+=1


def ML_mdoel(X,Y,model,X_test=False,para_path='para_path',file_name="",save=True):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            X: ndarray
                input values to train the model
            Y: ndarray
                output values corresponding to X
            model: str,default='decision tree'
                model name used for training, and more models will be added in future work
            X_test: ndarray
                input values to test the model
            para_path: str
                the directory for saving parameters of models during training
            file_name:
                the file name of the saved parameter file
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            output: ndarray
                the predicted values corresponding to X_test
            Examples
            --------

        """
    if model=='decision tree':
        regressor = DecisionTreeRegressor(random_state=0, max_features='auto')
        depths = np.arange(15, 41,5)
        num_leafs = np.arange(1, 10, 1)
        min_impurity_split = [pow(10, -30), pow(10, -50), pow(10, -80)]  # pow(10, -10),
        param_grid = [{'max_depth': depths,
                       'min_samples_leaf': num_leafs,
                       'min_impurity_split': min_impurity_split
                       }]
        gs = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)
        gs = gs.fit(X, Y)
        my_model = gs.best_estimator_
        my_model.fit(X, Y)
        if save:
            filename = para_path+'/'+file_name
            pickle.dump(my_model, open(filename, 'wb'))
            my_model = pickle.load(open(filename, 'rb'))

        if X_test==False:
            X_test=X
        Y_predict = my_model.predict(X_test)
    return Y_predict
def Relative_Error(Y,Y_predict):
    return np.mean(np.abs(Y-Y_predict)/np.abs(Y))
def test():
    root_dir = 'data_sample/'#the directory of all required climate variable files, for example: root_dir = 'D:/lifa/lawrence/code/Causality/datasets/'
    file_variables = {'Temperature.nc': 'temperature'}# the file name and its corresponding variable name, for example:  file_variables = {'Temperature.nc': 'temperature','Precipitation.nc':'precipitation'}
    time_lag = 2 # time delay used for modeling
    time_dim, lat_dim, lon_dim = 0, 2, 3 # the index of dimensions, for example: for a variable with shape（600，15，192，288）, the 0 dim (600) is time, 2 dim (192) is lat
    vars = data_loading(root_dir, file_variables)
    target_idx = [0]# list of indexes for predicted variables, for example: if file_variables = {'Temperature.nc': 'temperature','Precipitation.nc':'precipitation'}, target_idx = [0] means to predict next monthly temperature
    train(vars, target_idx, time_lag, time_dim, lat_dim, lon_dim)
if __name__=='__main__':
    test()