
from sklearn import set_config
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from flaml import AutoML
from common_func import evaluate_method, read_data, save_result

import pandas as pd

# X, y, GeoID = read_data.read_data_ID('test_data_wanzhou.csv')
def data_raw():
    train = pd.read_csv('test_data_wanzhou.csv')
    target = 'class'
    IDCol = 'OBJECTID'
    GeoID = train[IDCol]
    print(train[target].value_counts())
    # x_columns = [x for x in train.columns if x not in [target,IDCol,'GRID_CODE']]
    x_columns = ['Elevation', 'Slope', 'Aspect', 'TRI', 'Curvature', 'Lithology', 'River', 'NDVI', 'NDWI', 'Rainfall', 'Earthquake', 'Land_use']
    # x_columns = ['Elevation', 'Slope', 'Aspect', 'TRI', 'Curvature', 'Lithology', 'River', 'NDVI', 'NDWI', 'Rainfall']

    # X_colums = [x for x in train.columns if x not in [target,IDCol,'GRID_CODE']]
    X = train[x_columns]
    y = train[target]
    return X, y, GeoID
X, y, GeoID = data_raw()
X_train, X_test, y_train, y_test =sklearn.model_selection.train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)

set_config(display='diagram') ## 展示Piline图
imputer = SimpleImputer()
standardizer = StandardScaler()
automl = AutoML()

automl_pipeline = Pipeline([
    ("imputuer",imputer),
    ("standardizer", standardizer),
    ("automl", automl)
])

## 自动建模设置
settings = {
    "time_budget": 60,  # total running time in seconds
    "metric": 'f1',  # primary metrics can be chosen from: ['accuracy','roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'f1','log_loss','mae','mse','r2']
    "task": 'classification',  # task type   
    "estimator_list":['extra_tree'],
    "log_file_name": 'airlines_experiment.log',  # flaml log file
}

## 自动建模，管道化操作
automl_pipeline.fit(X_train, y_train, 
                        automl__time_budget=settings['time_budget'],
                        automl__metric=settings['metric'],
                        automl__estimator_list=settings['estimator_list'],
                        automl__log_training_metric=True)
print(automl_pipeline)

## 从Pipeline对象中提取 automl 对象
automl = automl_pipeline.steps[2][1]

## 
print('Best ML leaner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

automl.model

import pickle
with open('automl.pkl', 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

y_pred = automl_pipeline.predict(X)
print('Predicted labels', y_pred)
print('True labels', y_test)
y_pred_proba = automl_pipeline.predict_proba(X)[:,1]
print('Predicted probas ',y_pred_proba[:5])
evaluate_method.plotROC_1D(y_pred_proba, y, plotROC=True)

result_file_automl = './result/automl.txt'
save_result.save_ID_Class_prob(GeoID, y_pred, y_pred_proba, result_file_automl)