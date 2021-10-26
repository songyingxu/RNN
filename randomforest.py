from imblearn.ensemble import (BalancedRandomForestClassifier)
from common_func import evaluate_method, read_data, save_result

train_x, train_y_1D,_ = read_data.read_data_ID('train_data_wanzhou.csv')
test_x, test_y_1D, GeoID = read_data.read_data_ID('test_data_wanzhou.csv')

wrf = BalancedRandomForestClassifier(random_state=0)

wrf.fit(train_x, train_y_1D)
y_pred_wrf = wrf.predict(test_x)
y_pred_wrf_p = wrf.predict_proba(test_x)[:, 1]

evaluate_method.plotROC_1D(y_pred_wrf_p, test_y_1D, plotROC=True)

result_file_wlr = './result/wlr.txt'
save_result.save_ID_Class_prob(GeoID, y_pred_wrf, y_pred_wrf_p, result_file_wlr)

