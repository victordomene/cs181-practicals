import os
import cPickle as pickle
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import xgboost as xgb
import util

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

    print "# Done!"

if __name__ == "__main__":
    main()
    
