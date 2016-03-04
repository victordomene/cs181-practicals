# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
import re
import cPickle as pickle
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import util
import xgboost as xgb

def main():
    X_train, t_train, _ = pickle.load(open("../../features/all_tags/train.pickle"))
    X_test, _, test_ids = pickle.load(open("../../features/all_tags/test.pickle"))

    dtrain = xgb.DMatrix(X_train, label=t_train)

    param = {'bst:max_depth':30, 'eta':0.1, 'silent':2, 'objective':'multi:softprob', 'num_class': 15 }
    param['eval_metric'] = 'merror'
    param['min_child_weight'] = 3
    param['nthread'] = 16
    param['colsample_bytree'] = 0.5
    evallist = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, 1000, evallist)

    dout = xgb.DMatrix(X_test)
    t_probs = bst.predict(dout)
    t_pred = [prob.tolist().index(max(prob)) for prob in t_probs]

    util.write_predictions(t_pred, test_ids, "predictions.csv")

if __name__ == "__main__":
    main()
    