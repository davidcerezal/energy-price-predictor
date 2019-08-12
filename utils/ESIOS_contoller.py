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
            data_consumo = pd.read_csv("/content/drive/My Drive/TFM/Utils/data/"+name)
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
    
    def get_data_real_time(self):
        """
        Get the current real time available data and returns in dataframe format
        :return:
        """
        data_consumo_temp = self.data.drop(columns=['PVPC_DEF',
            'PVPC_2_PED_NOC',
            'PVPC_ELEC_NOC',
            'Precio mercado SPOT Diario_x',
            'Precio SPOT PT',
            'Precio SPOT FR',
            'Demanda real',
            'fecha',
            'date_day',
            'date_timestamp'                                  
        ]).copy()
        
        return data_consumo_temp   
   
    def get_target_data(self):
        """
        Get the current target available data and returns in dataframe format
        :return:
        """
        data_consumo_temp = self.data['PVPC_DEF'].copy()
        
        return data_consumo_temp 
            

        



