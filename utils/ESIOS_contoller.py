"""
ESIOS: Main class to access the Spanish electricity market data in dataframe format.

Copyright 2019 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import json
import urllib
import pandas as pd
import numpy as np
import datetime
import pickle
import requests
import json
from sklearn.utils import check_array
from scipy.stats import *
from sklearn.metrics import *
from math import sqrt
from keras.models import Model, model_from_json


class ESIOS(object):
    
    def __init__(self, in_colab):
        """
        Class constructor
        """
        import numpy as np
        self.token = '874b2aeaf4071f8db7085e0ca6aa88ede7e2992360fc0d6ad7af625b30b9e3bf'
        self.in_colab = in_colab


    def get_data(self, format=None):
        """
        Get the current available data and returns in dataframe format
        :return:
        """
        if format =='non-secuencial':
            name = 'data_total_for_non_serial.csv'
        else:    
            name = 'data_total.csv'
            
        if self.in_colab:
            data_consumo = pd.read_csv("/content/drive/My Drive/TFM/01.Utils/data/"+name)
        else:
            data_consumo = pd.read_csv("../utils/data/"+name)
            
        data_consumo = data_consumo.loc[:, ~data_consumo.columns.str.contains('^Unnamed')]
        data_consumo = data_consumo.fillna(method='ffill')
        data_consumo = data_consumo.fillna(method='bfill')
        
        self.data = data_consumo
            
        print('Mostrando los datos de '+name)
        print(data_consumo.shape)
        print('_'*80)
        data_consumo.head()
        
        return data_consumo  
      
    def get_df_daily(self):
        """
        Parse data to daily with mean()
        :return:
        """
        x_data_grouped = self.data.copy()
        x_data_grouped = x_data_grouped.set_index('fecha')
        x_data_grouped.index = pd.to_datetime(x_data_grouped.index, utc=True)
        x_data_grouped = x_data_grouped.groupby(pd.Grouper(freq='D')).mean()
        return x_data_grouped
     
    def get_df_daily_all_day_prices(self):
        """
        Parse data to daily and return for each day the 24 prices
        :return:
        """
        x_data_grouped = self.data.copy()
        x_data_grouped = x_data_grouped.set_index('fecha')
        x_data_grouped.index = pd.to_datetime(x_data_grouped.index, utc=True)
        x_data_grouped = x_data_grouped.groupby(pd.Grouper(freq='D'))
        y_data_encoder = x_data_grouped['PVPC_DEF'].apply(list)
        return y_data_encoder    
      
    def get_df_daily_target_day_prics(self):
        """
        Parse data to daily and return for each day the 24 target prices
        :return:
        """
        x_data_grouped = self.data.copy()
        x_data_grouped = x_data_grouped.set_index('fecha')
        x_data_grouped.index = pd.to_datetime(x_data_grouped.index, utc=True)
        x_data_grouped = x_data_grouped.groupby(pd.Grouper(freq='D'))
        y_data_encoder = x_data_grouped['PVPC-target'].apply(list)
        return y_data_encoder 
      
    def get_metrics(self, y_real, y_pred):
        """
       	Metrics
        :return:
        """
        print('**', 15*'-','Metrics:',15*'-' , '**')
        mse = mean_squared_error(y_real, y_pred)
        print('MSE: ', mse)
        rmse = sqrt(mean_squared_error(y_real, y_pred))
        print('RMSE: ', rmse)
        mae = mean_absolute_error(y_real, y_pred)
        print('MAE: ', mae)
        log_mse = mean_squared_log_error(y_real, y_pred)
        print('Log_MSE: ', log_mse)
        med_ae = median_absolute_error(y_real, y_pred)
        print('MedianAE: ', med_ae)
        mape = np.mean(abs((y_real-y_pred)/y_real))*100
        print('MAPE: ', mape, '%')
        smape = np.mean((np.abs(y_pred - y_real) * 200/ (np.abs(y_pred) + np.abs(y_real))))
        print('SMAPE: ', smape, '%')
        explained_variance = explained_variance_score(y_real, y_pred)
        print('Variance: ', explained_variance)
        max_error_var = max_error(y_real, y_pred)
        print('Max Error: ', max_error_var)
        print('**', 40*'-', '**')
        

    def get_data_real_time(self):
        """
        Get the current real time available data and returns in dataframe format
        :return:
        """
        data_consumo_temp = self.data.drop(columns=[
            'PVPC-target',
            'fecha',
            'date_day',
            'date_timestamp'                                  
        ]).copy()
        
        return data_consumo_temp   
      
    def get_selected_data(self):
        """
        Get the current real time available data and returns in dataframe format
        :return:
        """
        data_consumo_temp = self.data.copy()
        data_consumo_temp = data_consumo_temp.drop(columns=[
            'PVPC-target',
            'fecha',
            'date_day',
            'Programada',
            'Prevista',
            'Demanda real',
            'Precio mercado SPOT Diario_x',
            'Saldo total interconexiones programa p48',
            'Nuclear',
            'Solar',
            'Solar_Fotovoltaica',
            'Generación prevista Solar',
        ])
        data_consumo_temp = data_consumo_temp.set_index('date_timestamp')
        self.data = data_consumo_temp
        
        return data_consumo_temp        
      
   
    def get_target_data(self):
        """
        Get the current target available data and returns in dataframe format
        :return:
        """
        data_consumo_temp = self.data[['PVPC-target']].copy()
        
        return data_consumo_temp 
            

    def mean_absolute_percentage_error(self, y_pred, y_true ): 
        y_true, y_pred = check_array(y_true, y_pred)
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
      
    def save_keras_model(self, export_path, model, model_name, create_folder=False, save_weights=False):
        try:
            if create_folder:
                aux = model_name.split('_')[-1]
                assert aux not in os.listdir(export_path), 'Directory "{0}" already created.'.format(aux)
                os.mkdir(export_path + "/" + aux)
                model_path = export_path + "/" + aux

            else:
                model_path = export_path

            model_json = model.to_json()

            with open(model_path + "/" + model_name + ".json", "w") as json_file:
                json_file.write(model_json)

            if save_weights:
                model.save_weights(model_path + "/" + model_name + ".h5")

            print("Saved model to disk as {0}/{1}".format(model_path, model_name))
            return True, model_path

        except Exception as err:
            print('Save error: {0}'.format(err))
            return False, err


    def load_keras_model(self, import_path):
        try:
            path = import_path + ".json" if not import_path.endswith(".json") else import_path
            print(path)

            with open(path, "r") as json_file:
                loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(path.replace('.json', '.h5'))

            print("Loaded model from disk")
            return loaded_model

        except Exception as err:
            print("Loading error: {0}".format(err))
            return False




