#!/usr/bin/python
# -*- coding=utf-8 -*-
'''
=======================================================
Read landslide data from csv file
=======================================================

'''

import pandas as pd
import numpy as np


def read_data(file_name):
    data = pd.read_csv(file_name)
    names = data.columns
    data = data.values
    data_x = data[:,:-1]
    data_y = data[:,-1]
    return data_x, data_y

def read_data_ID(file_name):
    data = pd.read_csv(file_name)
    print(data['class'].value_counts())
    GEOID = data['OBJECTID']
    data = data.values
    data_x = data[:,1:-1]
    data_y = data[:,-1]
    return data_x, data_y, GEOID
