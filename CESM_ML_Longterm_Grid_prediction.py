"""Use Machine Learning (ML) methods to surrogate traditional climate models, and make predictions.
Author = Fa Li

.. MIT License
..
.. Copyright (c) 2019 Fa Li
"""
from netCDF4 import Dataset
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import os
import time
import math
from sklearn.metrics import explained_variance_score
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def data_loading(data_dir,file_variables):#
    """**Description:** this method is used to load the required climate variables

        Parameters
        ----------
        data_dir: str,default=""
            the directory of all files corresponding to required climate variables
        file_names: {}, default={}
            a dictionary mapping each file name to its corresponding variable name, the corresponding variable names are contained in a list
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
        nc_file = Dataset(file_path, 'r')
        var_names=file_variables[file]
        for var_name in var_names:
            var_value =nc_file[var_name][:]
            print(var_name,var_value.shape)
            if len(var_value.shape)==3:
                var_value=var_value[:,np.newaxis,:,:]
            output.append(var_value)
        nc_file.close()
    return output
def train(vars,target_idx,time_lag,time_dim,lat_dim,lon_dim,mask='no',time_lead=3,target_hist=False,long_term_strategy=1,para_path='para_path',save=True):
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
            time_lead: int
                how long to be predicted in advance
            target_hist: bool,default=False
                whether to include historial record of the target variable as input
            long_term_strategy: int, default=1
                if the value==1, each prediction at each time step is a model
                if the value==2, prediction at all time steps is the same model
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
    time_length = time_length - time_lead
    target_fea_num = len(target_idx) * time_lead
    input_fea_num=0
    for var_idx in range(len(vars)):
        input_fea_num+=vars[var_idx].shape[1]
    if not target_hist:
        input_fea_num-=1
    total_task_num = np.sum(mask==1) * target_fea_num
    task_idx=0
    predicted_results=np.full((time_length,target_fea_num,rows,cols),np.nan)
    importance_results=np.full((input_fea_num,target_fea_num,rows,cols),np.nan)
    if mask=='no':
        mask=np.ones((rows,cols))
    for lat in range(rows):
        for lon in range(cols):
            if mask[lat,lon]==1 and np.isnan(predicted_results[0,0,lat,lon]):
                input = np.ones((time_length, 1))
                output=np.zeros((time_length, target_fea_num))
                output_cnt=0
                for var_idx in range(len(vars)):
                    for level in range(vars[var_idx].shape[1]):
                        series=vars[var_idx][:, level,lat, lon]
                        for lag in range(1,time_lag+1):
                            temp=series[(time_lag-lag):((time_lag-lag)+time_length)]
                            temp=temp[:,np.newaxis]
                            if var_idx not in target_idx:
                                input=np.concatenate((input,temp),axis=1)
                            elif target_hist:
                                input = np.concatenate((input, temp), axis=1)
                        if var_idx in target_idx:
                            for lead in range(time_lead):
                                output[:,output_cnt]=series[(time_lag+lead):(time_lag+time_length+lead)]
                                output_cnt+=1
                input=input[:,1:input.shape[1]]
                if long_term_strategy==1:
                    for target in range(target_fea_num):
                        Y=output[:,target]
                        Y_predict,importances=ML_mdoel_LeaveOneOut(X=input,Y=Y,model='decision tree',file_name=str(lat)+'_'+str(lon),save=save)
                        print('relative error is:',Relative_Error(Y,Y_predict))
                        print('explained_variance_score:', explained_variance_score(Y,Y_predict))
                        print('pearsonr:',stats.pearsonr(Y,Y_predict)[0])
                        predicted_results[:,target,lat,lon]=Y_predict[:]
                        importance_results[:,target,lat,lon]=importances
                        if (task_idx + 1)%10==0:
                            print("processing:{0}%".format(round((task_idx + 1) * 100 / total_task_num,5)))
                            time.sleep(0.01)
                        task_idx+=1
                        if task_idx%10==0:
                            nc_save(predicted_results,importance_results)
                elif long_term_strategy==2:
                    Y=output
                    Y_predict, importances=ML_mdoel(X=input,Y=Y,model='decision tree')


    nc_save(predicted_results, importance_results)
def nc_save(predicted_results,importances):
    nc_file_path = r'C:\Users\lmars\Desktop\urgent\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    root_dir = r"C:\Users\lmars\Desktop\urgent\wildfire_prediction\ELM_fire\ELM_fire\input_features"
    nc_fid2 = Dataset(root_dir + '/regional_predicted_wildfire_3month_lead.nc', 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('fea', importances.shape[0])
    nc_fid2.createDimension('time_lead', importances.shape[1])
    nc_fid2.createDimension('time', predicted_results.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    features= nc_fid2.createVariable('features', 'f4', ('fea',))
    # types = nc_fid2.createVariable("climate_types", "f8", ("climate_types",))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    time_lead=nc_fid2.createVariable("time_lead", "f8", ("time_lead",))
    burntArea = nc_fid2.createVariable('burntArea', "f8", ("time","time_lead", "lat", "lon",))  # "climate_types",
    feature_importances=nc_fid2.createVariable('feature_importances', "f8", ("fea", "time_lead","lat", "lon",))
    time_v[:] = range(predicted_results.shape[0])
    time_lead[:]=range(importances.shape[1])
    features[:]=range(importances.shape[0])
    # types[:] = range(Model_input_Obs.shape[1])
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    burntArea[:] = predicted_results[:]
    feature_importances[:]=importances[:]
    nc_fid2.close()

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
def ML_mdoel_LeaveOneOut(X,Y,model,delta_time=1,para_path='para_path',file_name="",save=False):
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
        regressor = DecisionTreeRegressor(random_state=0, max_features='auto',max_depth=150)
        #depths = np.arange(50, 100,20)
        num_leafs = np.arange(2, 5, 1)
        #min_impurity_split = [pow(10, -30), pow(10, -50), pow(10, -80)]  # pow(10, -10),
        param_grid = [{#'max_depth': depths,
                       #'min_impurity_split': min_impurity_split,
                       'min_samples_leaf': num_leafs
                       }]
        gs = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error',cv=5)#,
        all_indxs=[idx for idx in range(X.shape[0])]
        Y_predict=np.zeros(Y.shape)
        for sample_idx in range(0,X.shape[0],delta_time):
            if sample_idx+delta_time<=X.shape[0]:
                end_idx=sample_idx+delta_time
            else:
                end_idx=X.shape[0]
            sample_idxs=[idx for idx in range(sample_idx,end_idx)]
            train_idxs=list(set(all_indxs).difference(set(sample_idxs)))
            X_input=X[train_idxs]
            Y_input=Y[train_idxs]
            X_test=X[sample_idxs]
            gs = gs.fit(X_input, Y_input)
            my_model = gs.best_estimator_
            my_model.fit(X_input, Y_input)
            importances = my_model.feature_importances_
            if save:
                filename = para_path+'/'+file_name
                pickle.dump(my_model, open(filename, 'wb'))
                my_model = pickle.load(open(filename, 'rb'))
            Y_predict[sample_idxs] = my_model.predict(X_test)
    if model=='random forest':
        regressor = RandomForestRegressor(random_state=0, max_features='sqrt',oob_score=True,n_jobs=4)
        depths = np.arange(15, 100,20)
        num_leafs = np.arange(1, 6, 2)
        n_estimators=np.arange(5,20,5)
        #min_impurity_split = [pow(10, -30), pow(10, -50), pow(10, -80)]  # pow(10, -10),
        param_grid = [{'max_depth': depths,
                       'min_samples_leaf': num_leafs,
                       #'min_impurity_split': min_impurity_split,
                       'n_estimators':n_estimators
                       }]
        gs = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error')#, cv=10
        all_indxs=[idx for idx in range(X.shape[0])]
        Y_predict=np.zeros(Y.shape)
        for sample_idx in range(0,X.shape[0],12):
            if sample_idx+12<=X.shape[0]:
                end_idx=sample_idx+12
            else:
                end_idx=X.shape[0]
            sample_idxs=[idx for idx in range(sample_idx,end_idx)]
            train_idxs=list(set(all_indxs).difference(set(sample_idxs)))
            X_input=X[train_idxs]
            Y_input=Y[train_idxs]
            X_test=X[sample_idxs]
            gs = gs.fit(X_input, Y_input)
            my_model = gs.best_estimator_
            my_model.fit(X_input, Y_input)
            if save:
                filename = para_path+'/'+file_name
                pickle.dump(my_model, open(filename, 'wb'))
                my_model = pickle.load(open(filename, 'rb'))
            Y_predict[sample_idxs] = my_model.predict(X_test)
    return Y_predict,importances
def ML_mdoel_LeaveOneOut_Regions(X,Y,model,region_mask,para_path='region_para_path',save=False):
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
    delta_time=12
    region_num=14

    if model=='decision tree':
        model_gs={}
        for model_idx in range(1,(1+region_num)):
            regressor = DecisionTreeRegressor(random_state=0, max_features='auto',max_depth=150)
            #depths = np.arange(50, 100,20)
            num_leafs = np.arange(3, 16, 2)
            #min_impurity_split = [pow(10, -30), pow(10, -50), pow(10, -80)]  # pow(10, -10),
            param_grid = [{#'max_depth': depths,
                           #'min_impurity_split': min_impurity_split,
                           'min_samples_leaf': num_leafs
                           }]
            gs = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error',cv=5)
            model_gs[model_idx]=gs
        all_indxs=[idx for idx in range(X.shape[0])]
        target_shape=Y.shape
        Y_predict=np.full(target_shape,np.nan)
        result_importance=np.full(X.shape,np.nan)
        task_idx=0
        total_task_num=math.ceil(X.shape[0]/delta_time)*region_num
        for sample_idx in range(0,X.shape[0],delta_time):#start from X.shape[0]-delta_time to get well-trained trees
            if sample_idx+delta_time<=X.shape[0]:
                end_idx=sample_idx+delta_time
            else:
                end_idx=X.shape[0]
            sample_idxs=[idx for idx in range(sample_idx,end_idx)]
            train_idxs=list(set(all_indxs).difference(set(sample_idxs)))
            X_input=X[train_idxs]#(time_length,48,96,144)
            Y_input=Y[train_idxs]#(time_length,96,144)
            X_test=X[sample_idxs]#(time_length,48,96,144)
            for region_idx in range(1,(1+region_num)):
                X_merge_input=np.ones((1,X_input.shape[1]))
                Y_merge_input=np.array([1])
                for lat in range(X_input.shape[2]):
                    for lon in range(X_input.shape[3]):
                        if region_mask[lat,lon]==region_idx and not(np.isnan(X_input[0,0,lat,lon])):
                            X_merge_input=np.concatenate((X_merge_input,X_input[:,:,lat,lon]),axis=0)
                            Y_merge_input=np.concatenate((Y_merge_input,Y_input[:,lat,lon]),axis=0)
                if X_merge_input.shape[0]>1:
                    X_merge_input=X_merge_input[1:X_merge_input.shape[0]]
                    Y_merge_input=Y_merge_input[1:Y_merge_input.shape[0]]
                print('start training of region:',region_idx)
                if X_merge_input.shape[0]>100:
                    gs = model_gs[region_idx].fit(X_merge_input, Y_merge_input)
                    my_model = gs.best_estimator_
                    my_model.fit(X_merge_input, Y_merge_input)
                    importances = my_model.feature_importances_
                    if save:
                        filename = para_path + '/'  +str(region_idx)
                        pickle.dump(my_model, open(filename, 'wb'))
                        #my_model = pickle.load(open(filename, 'rb'))
                    print(region_idx,importances)
                    for lat in range(X_input.shape[2]):
                        for lon in range(X_input.shape[3]):
                            if region_mask[lat,lon]==region_idx and not(np.isnan(X_test[0,0,lat,lon])):
                                Y_predict[sample_idxs,lat,lon]=my_model.predict(X_test[:,:,lat,lon])
                                for min_idx in sample_idxs:
                                    result_importance[min_idx,:,lat,lon]=importances
                #task_idx+=1
                if (task_idx + 1) % 10 == 0:
                    print("processing:{0}%".format(round((task_idx + 1) * 100 / total_task_num, 5)))
                    time.sleep(0.01)
                task_idx += 1

            #Y_predict[sample_idxs] = my_model.predict(X_test)
        importances=np.mean(result_importance,axis=0)
    return Y_predict,importances
def load_mask():
    nc_file_path = r'C:\Users\lmars\Desktop\urgent\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    temp = data.variables['TSA'][:]
    data.close()
    nc_file_path = r'C:\Users\lmars\Desktop\urgent\wildfire_prediction\ELM_fire\ELM_fire\input_features\wildfire_9506-1012.nc'
    data = Dataset(nc_file_path, 'r')
    wildfires = data.variables['burntArea'][:]
    data.close()
    temp=temp[0]
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            value=temp[i,j]
            zero_cnt=wildfires[:,i,j].tolist().count(0)
            if math.isnan( float(value)) or zero_cnt>136:
                temp[i,j]=0
            else:temp[i,j]=1
    print('wildfire area:',np.sum( temp == 1 ) )
    plt.imshow(temp)
    plt.show()
    return temp

def Relative_Error(Y,Y_predict):
    return np.median(np.abs(Y-Y_predict)/(np.abs(Y)+pow(10,-5)))
def load_region():
    nc_file_path = r'C:\Users\lmars\Desktop\urgent\wildfire_prediction\ELM_fire\basic_14regions.nc'
    data = Dataset(nc_file_path, 'r')
    regions = data.variables['basic_14regions'][:]
    data.close()
    regions[regions==0]=np.nan
    plt.imshow(regions)
    plt.show()
    return regions
def test():
    root_dir = r'C:\Users\lmars\Desktop\urgent\wildfire_prediction\ELM_fire\ELM_fire\input_features/'#the directory of all required climate variable files, for example: root_dir = 'D:/lifa/lawrence/code/Causality/datasets/'
    file_variables = {'wildfire_9506-1012.nc':['burntArea'],#187x96x144
                    'lnfm.nc': ['lnfm'],#12 values: 1-12 month#12 monthsx96x144
                      'populationDensity_95-10.nc':['hdm'],#16 yearsx96x144
                      'landuse_years_95-10.nc':['PCT_NAT_PFT'],#16 yearsx16typesx96x144
                      # land use type: https://daac.ornl.gov/VEGETATION/guides/fire_emissions_v4.html
                      'ELM.fire.sub.nc':['TSA','RAIN','FSDS','FLDS','WIND','RH2M','PBOT',
                                         'PCT_NAT_PFT',#monthly records
                                         'SOILWATER_10CM',
                                         'CWDC','TOTVEGC','TOTLITC']}# the file name and its corresponding variable name, for example:  file_variables = {'Temperature.nc': 'temperature','Precipitation.nc':'precipitation'}
    time_lag = 1 # time delay used for modeling
    time_lead = 3 # how long to be predicted in advance
    time_dim, lat_dim, lon_dim = 0, 2, 3 # the index of dimensions, for example: for a variable with shape（600，15，192，288）, the 0 dim (600) is time, 2 dim (192) is lat
    vars = data_loading(root_dir, file_variables)
    print(len(vars))
    target_idx = [0]# list of indexes for predicted variables, for example: if file_variables = {'Temperature.nc': 'temperature','Precipitation.nc':'precipitation'}, target_idx = [0] means to predict next monthly temperature
    mask=load_mask()# mask is used to filter out those specified areas for prediction
    train(vars, target_idx, time_lag, time_dim, lat_dim, lon_dim,mask,time_lead)
if __name__=='__main__':
    test()