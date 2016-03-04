import os
import cPickle as pickle
import random
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(util.malware_classes))
    plt.xticks(tick_marks, util.malware_classes, rotation=45)
    plt.yticks(tick_marks, util.malware_classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    # fetch features for training and valid data
    # substitute for a pickle load, for training data!
    print "# Loading features..."
    X_train, t_train, _ = pickle.load(open("../../features/all_tags/train.pickle"))

    dtrain = xgb.DMatrix(X_train, label=t_train)

    param = {'bst:max_depth':30, 'eta':0.1, 'silent':2, 'objective':'multi:softprob', 'num_class': 15 }
    param['eval_metric'] = 'merror'
    param['min_child_weight'] = 3
    param['nthread'] = 16
    param['colsample_bytree'] = 0.5
    evallist = [(dtrain,'train')]

    print "# Running Cross Validation with nfold = 5..."
    scores = xgb.cv(param, dtrain, 500, nfold=5,
       metrics={'merror'}, seed = 0, show_stdv = False)

    print scores

if __name__ == "__main__":
    main()
    
