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
    X_train_all, t_train_all, train_all_ids = pickle.load(open("../../features/all_tags/train.pickle"))
    
    print "# Separating features into two simple sets..."
    X_train, t_train, train_ids = X_train_all[:2000], t_train_all[:2000], train_all_ids[:2000]
    X_test, t_test, test_ids = X_train_all[2000:], t_train_all[2000:], train_all_ids[:2000]

    print "# Training data with XGBoost..."
    dtrain = xgb.DMatrix(X_train, label=t_train)

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

    print "# Plotting confusion matrix..."
    # compute and plot confusion matrix
    confmat = confusion_matrix(t_test + 1, np.array(t_pred) + 1)
    cm_normalized = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)
    print('Confusion matrix, normalized')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized)
    # plt.show()
    plt.savefig("confmatrix_normalized.png")

    count = 0
    correct = 0
    for true, pred in zip(t_test, t_pred):
        count += 1
        if true == pred:
            correct += 1

    print "# Percentage of correct classifications:"
    print float(correct)/count * 100

if __name__ == "__main__":
    main()
    
