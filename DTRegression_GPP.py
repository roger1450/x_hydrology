from netCDF4 import Dataset
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
root_dir='D:/lifa/lawrence/code/Causality/datasets/'
nc_file_path=root_dir+'SIF_2_normal_moving_masked2.nc'
data=Dataset(nc_file_path,'r')
SIF=data.variables['sif_normal'][:]
mask=data.variables['sif_normal'][:]
mask=mask[0,0]
nan_value=mask[0,0]
lon=data.variables['longitude'][:]
lat=data.variables['latitude'][:]
###load test data
nc_file_path=root_dir+'/Temperature_2.nc'
data=Dataset(nc_file_path,'r')
T_Obs=data.variables['temperature'][:]#gpp,pr,tas,rsds
nc_file_path=root_dir+'/prcp_2.nc'
data=Dataset(nc_file_path,'r')
prcp_Obs=data.variables['prcp'][:]#gpp,pr,tas,rsds
nc_file_path=root_dir+'/dswrf_2.nc'
data=Dataset(nc_file_path,'r')
R_Obs=data.variables['dswrf'][:]#

nino_obs_files=['nino3_SST_HadISST.data.txt','nina34_SST_ersstv5.data.txt','nina4_SST_ersstv5.data.txt']
nino_data_Obs=np.zeros((324,3))
nino_time_index=[110,32,32,136,58,58]
for i in range(3):
    temp_data=np.loadtxt(root_dir+nino_obs_files[i])[(nino_time_index[i]-1):nino_time_index[i+3],1:13]
    nino_data_Obs[:, i] =temp_data.reshape(-1)+273.15
time_lag=8
var_num=4

####load train data
subfixs=['__bcc-csm1-1-m_esmHistorical_r1i1p1_185001-201212','__inmcm4_esmHistorical_r1i1p1_185001-200512','__BNU-ESM_esmHistorical_r1i1p1_185001-200512',
        '__MIROC-ESM_esmHistorical_r1i1p1_185001-200512','__IPSL-CM5A-LR_esmHistorical_r1i1p1_185001-200512','__CESM1-BGC_esmHistorical_r1i1p1_185001-200512',
        '__GFDL-ESM2G_esmHistorical_r1i1p1_186101-200512','__CanESM2_esmHistorical_r1i1p1_185001-200512','__HadGEM2-ES_esmHistorical_r1i1p1_185912-200512',
        '__NorESM1-ME_esmHistorical_r1i1p1_185001-200512'
        ]
MSE_Result={}
RE_Result={}
df = pd.DataFrame(columns = ["model",'gpp_re','gpp_rmse'])
idx = 0
for subfix in subfixs:
    model_name = subfix.split('_')[2]
    nc_file_path=root_dir+'/DATA/tas_2'+subfix+'.nc'
    data=Dataset(nc_file_path,'r')
    T=data.variables['tas'][:]#gpp,pr,tas,rsds
    nc_file_path=root_dir+'/DATA/pr_2'+subfix+'.nc'
    data=Dataset(nc_file_path,'r')
    prcp=data.variables['pr'][:]#gpp,pr,tas,rsds
    nc_file_path=root_dir+'/DATA/rsds_2'+subfix+'.nc'
    data=Dataset(nc_file_path,'r')
    R=data.variables['rsds'][:]#gpp,pr,tas,rsds
    nc_file_path = root_dir + '/DATA/gpp_2' + subfix + '.nc'
    data = Dataset(nc_file_path, 'r')
    gpp = data.variables['gpp'][:]
    train_out_input=np.zeros((600-time_lag,time_lag*var_num+1))
    test_out_input=np.zeros((324-time_lag,time_lag*var_num+1))
    Model_input_Obs=np.full((324-time_lag,prcp_Obs.shape[2],prcp_Obs.shape[3]),np.nan)
    vars=[gpp,prcp,T,R]
    vars_Obs=[gpp,prcp_Obs,T_Obs,R_Obs]
    RMSE_Model=[[],[]]
    RE=[]

    #RMSE_Model_Obs=[[],[],[]]
    for i in range(32,57):
                    for j in range(30,75):
                        if np.isnan(SIF[:,0,i,j]).sum()==0 and np.isnan(gpp[:,0,i,j]).sum()==0 and np.isnan(prcp[:,0,i,j]).sum()==0 and np.isnan(T[:,0,i,j]).sum()==0 and np.isnan(R[:,0,i,j]).sum()==0:
                            #train process
                            train_out_input[:, 0] = vars[0][time_lag:600, 0, i, j]
                            test_out_input[:, 0] = vars_Obs[0][(600-324+time_lag):600, 0, i, j]
                            for time_idx in range(time_lag):
                                train_out_input[:, (time_idx + 1)] = vars[0][(time_lag - time_idx - 1):(600 - time_idx - 1),
                                                                                          0, i, j]
                                test_out_input[:,  (time_idx + 1)] = vars_Obs[0][(600-324+time_lag - time_idx - 1):(600 - time_idx - 1),
                                                                                         0, i, j]
                            for var_idx in range(1, var_num):
                                for time_idx in range(time_lag):
                                    train_out_input[:, var_idx * time_lag + (time_idx + 1)] = vars[var_idx][(time_lag - time_idx - 1):(600 - time_idx - 1), 0, i, j]
                                    test_out_input[:, var_idx * time_lag + (time_idx + 1)] = vars_Obs[var_idx][(time_lag - time_idx - 1):( 324 - time_idx - 1), 0, i, j]

                            X=train_out_input[:,1:train_out_input.shape[1]]
                            Y=train_out_input[:,0]
                            X_test = test_out_input[:, 1:test_out_input.shape[1]]
                            Y_Obs = test_out_input[:, 0]
                            Y_model = Y[(Y.shape[0] - (324 - time_lag)):Y.shape[0]]
                            ##################################################################
                            #decision tree modeling
                            ########grid search Cross validation
                            regressor = DecisionTreeRegressor(random_state=0, max_features='auto')
                            #X_train_and_validate, X_test, y_train_and_validate, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
                            depths = [20, 30, 50, 80]  # np.arange(15, 41,5)
                            num_leafs = np.arange(1, 10, 1)
                            min_impurity_split = [pow(10, -30),  pow(10, -50), pow(10, -80)]#pow(10, -10),
                            param_grid = [{'max_depth': depths,
                                           'min_samples_leaf': num_leafs,
                                           'min_impurity_split':min_impurity_split
                                           }]
                            gs = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)
                            gs=gs.fit(X, Y)
                            my_model = gs.best_estimator_
                            my_model.fit(X, Y)
                            X_test=X[(Y.shape[0] - (324 - time_lag)):Y.shape[0]]
                            Y_predict = my_model.predict(X_test)
                            # decision tree modeling
                            ########grid search Cross validation
                            """
                            regressor = DecisionTreeRegressor(random_state=0,max_features='auto',max_depth=35,min_impurity_split=pow(10,-30))#30,
                            regressor.fit(X, Y)
                            Y_predict = regressor.predict(X_test)
                        """

                            """
                            plt.plot(range(X_test.shape[0]), Y_model, color='red',label='model')
                            plt.plot(range(X_test.shape[0]), Y_predict, color='blue',label='model_pred')
                            #plt.plot(range(X_test.shape[0]), Y_model, color='green', label='model')
                            plt.legend()
                            plt.show()
                        """
                            print('input-Obs RMSE:', np.sqrt(metrics.mean_squared_error(Y_Obs, Y_predict)))
                            print('input-Obs RE:', np.median(np.abs((Y_Obs- Y_predict)/(Y_Obs+pow(10,-10)))))
                            RE.append(np.median(np.abs((Y_Obs- Y_predict)/(Y_Obs+pow(10,-10)))))
                            RMSE_Model[0].extend(Y_Obs)
                            RMSE_Model[1].extend(Y_predict)
                            #RMSE_Model_Obs[var_idx][0].append(Y_Obs)
                            #RMSE_Model_Obs[var_idx][1].append(Y_predict)
                            #RMSE_Model_Obs[var_idx].append(np.sqrt(metrics.mean_squared_error(Y_Obs, Y_predict)))
                            Model_input_Obs[:, i, j] = Y_predict
                                #export_graphviz(regressor, out_file='tree.dot')
    Temp_MSE_Result=np.abs((np.array(RMSE_Model[0])-np.array(RMSE_Model[1]))/np.array(RMSE_Model[0]))
    MSE_Result[model_name]=Temp_MSE_Result
    print('input-Obs RE GPP:',Temp_MSE_Result)
    df.loc[idx] =[model_name,np.median(np.array(RE)),Temp_MSE_Result]
    idx+=1

    #np.save(root_dir+'/DATA/Model_pred/'+model_name+'_79-05.npy',Model_input_Obs)

    #save the predicted results as nc file
    nc_fid2 = Dataset(root_dir+'/DATA/Model_pred/GPP/'+model_name+'_DTModelInput_79-05.nc', 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    #nc_fid2.createDimension('climate_types', Model_input_Obs.shape[1])
    nc_fid2.createDimension('time', Model_input_Obs.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    #types = nc_fid2.createVariable("climate_types", "f8", ("climate_types",))
    time = nc_fid2.createVariable("time", "f8", ("time",))
    modelPred_Obs_input = nc_fid2.createVariable('DTPred_Model_input', "f8", ("time",  "lat", "lon",))#"climate_types",
    time[:] = range(Model_input_Obs.shape[0])
    #types[:] = range(Model_input_Obs.shape[1])
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    modelPred_Obs_input[:]=Model_input_Obs[:]
    nc_fid2.close()
df.to_csv('D:/lifa/lawrence/code/Causality/datasets/nino2ptr2gpp_csv_result/model_gpp_prediction_evaluation.csv')
f = open(root_dir+'/DATA/Model_pred/GPP/RMSE_MODEL_OBS_79-05.txt', 'w')
f.write(str(MSE_Result))
f.close()