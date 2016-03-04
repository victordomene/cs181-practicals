import os
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
    print "# Loading features..."
    X_train, t_train, _ = pickle.load(open("../../features/all_tags/train.pickle"))
    X_test, _, test_ids = pickle.load(open("../../features/all_tags/test.pickle"))

    dtrain = xgb.DMatrix(X_train, label=t_train)

    print "# Training XGBoost on training data..."
    param = {'bst:max_depth':30, 'eta':0.1, 'silent':2, 'objective':'multi:softprob', 'num_class': 15 }
    param['eval_metric'] = 'merror'
    param['min_child_weight'] = 3
    param['nthread'] = 16
    param['colsample_bytree'] = 0.5
    evallist = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, 500, evallist)

    print "# Predicting test data..."
    dout = xgb.DMatrix(X_test)
    t_probs = bst.predict(dout)
    t_pred = [prob.tolist().index(max(prob)) for prob in t_probs]

    util.write_predictions(t_pred, test_ids, "../../predictions/xgboost_predictions.csv")

    print "# Done!"

if __name__ == "__main__":
    main()
    
