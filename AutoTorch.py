import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier
from torch import nn

from common_func import evaluate_method, read_data, save_result

X, y, GeoID = read_data.read_data_ID('test_data_wanzhou.csv')
X = X.astype(np.float32)
y = y.astype(np.int64)
print(X.shape,y.shape)
X_train,X_test, y_train, y_test =sklearn.model_selection.train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)

class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(29, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X

from skorch.callbacks import Callback


def tweet(msg):
    print("~" * 60)
    print("*tweet*", msg, "#skorch #pytorch")
    print("~" * 60)


class AccuracyTweet(Callback):
    def __init__(self, min_accuracy):
        self.min_accuracy = min_accuracy

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        if self.critical_epoch_ > -1:
            return
        # look at the validation accuracy of the last epoch
        if net.history[-1, 'valid_acc'] >= self.min_accuracy:
            self.critical_epoch_ = len(net.history)

    def on_train_end(self, net, **kwargs):
        if self.critical_epoch_ < 0:
            msg = "Accuracy never reached {} :(".format(self.min_accuracy)
        else:
            msg = "Accuracy reached {} at epoch {}!!!".format(
                self.min_accuracy, self.critical_epoch_)

        tweet(msg)

net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=150,
    lr=0.02,
    warm_start=True,
    callbacks=[AccuracyTweet(min_accuracy=0.7)],
)

net.fit(X, y)