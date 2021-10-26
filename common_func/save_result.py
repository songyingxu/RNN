import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from keras.utils import np_utils
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

def save_ID_Class_prob(GeoID, y_pred, y_predprob, result_file):
    results = np.vstack((GeoID,y_pred,y_predprob))
    results = np.transpose(results)
    header_string = 'GeoID, y_pred, y_predprob'
    np.savetxt(result_file, results, header = header_string, fmt = '%d,%d,%0.5f',delimiter = ',')
    print('Saving file Done!')