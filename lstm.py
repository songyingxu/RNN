import random
import sklearn
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import (LSTM, Activation, BatchNormalization, Conv1D, Dense,
                          Flatten, MaxPool1D, SimpleRNN)
from keras.models import Sequential, load_model
from keras.utils import np_utils
from sklearn import metrics

from common_func import evaluate_method, loss_history, read_data, save_result

tf.random.set_seed(6)
np.random.seed(6)

def Prepare_Data(path):
    X, y, GeoID = read_data.read_data_ID(path)
    train_x, test_x, train_y_1D, test_y_1D = sklearn.model_selection.train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    train_y = np_utils.to_categorical(train_y_1D, 2)
    test_y = np_utils.to_categorical(test_y_1D, 2)

    train_x = np.expand_dims(train_x,axis=2)
    test_x = np.expand_dims(test_x,axis=2)
    return X,y,GeoID,train_x,test_x,test_y_1D,train_y,test_y

X, y, GeoID, train_x, test_x, test_y_1D, train_y, test_y = Prepare_Data('test_data_wanzhou.csv')

model = Sequential()
model.add(LSTM(50, batch_input_shape=(None, 29, 1), unroll=True))
# model.add(Dropout(0.5))
model.add(Dense(2))
# recurrent_activation = 'sigmoid'
model.add(Activation('sigmoid'))
optimizer = optimizers.adam_v2.Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Fit the model

print(model.summary())
history = loss_history.LossHistory()
model.fit(train_x,train_y,validation_data= (test_x,test_y),verbose=2,callbacks=[history],batch_size=32,epochs=100)

# model = load_model('my_model_lstm.h5')

y_pred_lstm = model.predict(test_x)   



#output predict probability
y_pred_lstm_p = [prob[1] for prob in y_pred_lstm]

evaluate_method.plotROC_1D(y_pred_lstm_p, test_y_1D, plotROC=True)



acc = evaluate_method.get_acc(test_y_1D, y_pred_lstm_p)  # AUC value
test_auc = metrics.roc_auc_score(test_y_1D,y_pred_lstm_p)
kappa = evaluate_method.get_kappa(test_y_1D, y_pred_lstm_p)
IOA = evaluate_method.get_IOA(test_y_1D, y_pred_lstm_p)
MCC = evaluate_method.get_mcc(test_y_1D, y_pred_lstm_p)
recall = evaluate_method.get_recall(test_y_1D, y_pred_lstm_p)
precision = evaluate_method.get_precision(test_y_1D, y_pred_lstm_p)
f1 = evaluate_method.get_f1(test_y_1D, y_pred_lstm_p)
# MAPE = evaluate_method.get_MAPE(test_y_1D,y_pred_lstm_p)

# evaluate_method.get_ROC(test_y_1D,y_pred_lstm_p,save_path='roc_lstm.txt')
print("ACC = " + str(acc))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))

model.save('my_model_lstm1.h5')
# history.loss_plot('epoch')

result_y = np_utils.to_categorical(y, 2)
result_x = np.expand_dims(X,axis=2)
y_pred = model.predict(result_x)
y_pred_proba = [prob[1] for prob in y_pred]

result_file_lstm = './result/lstm.txt'
save_result.save_ID_Class_prob(GeoID, y_pred_proba, y_pred_proba, result_file_lstm)