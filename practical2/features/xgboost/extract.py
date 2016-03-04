# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
import re
from collections import Counter
import random
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import util

import xgboost as xgb

def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x

counter = 0
TRAIN_DIR = "../../data/train/"
TEST_DIR = "../../data/test/"

train_shuffle_directory = shuffle(os.listdir(TRAIN_DIR))
test_shuffle_directory = shuffle(os.listdir(TEST_DIR))

call_set = set([])

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

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1

    dirs = []
    if direc == TRAIN_DIR:
        dirs = train_shuffle_directory
    else:
        dirs = test_shuffle_directory

    for datafile in dirs:
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)

        if clazz == "X":
            classes.append(-1)
        else:
            classes.append(util.malware_classes.index(clazz))

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids


def call_feats(tree):
    # keeping track of progress
    global counter
    counter += 1

    if counter % 100 == 0:
        print counter

    # all tags will be considered "good" for now
    good_calls = util.all_syscalls

    call_counter = {}
    for el in tree.iter():
        call = el.tag

        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]

    return np.array(call_feat_array, dtype=int)

## Feature extraction
def main():
    final_ids = []
    final_prediction = []

    # fetch features for training and valid data
    # substitute for a pickle load, for training data!

    X_train, t_train, train_ids = create_data_matrix(0, 2000, TRAIN_DIR)
    # X_valid, t_valid, valid_ids = create_data_matrix(2400, 2700, TRAIN_DIR)
    X_test, t_test, test_ids = create_data_matrix(2000, 4000, TRAIN_DIR)

    dtrain = xgb.DMatrix(X_train, label=t_train)
    # dtest = xgb.DMatrix(X_valid, label=t_valid)

    param = {'bst:max_depth':30, 'eta':0.1, 'silent':2, 'objective':'multi:softprob', 'num_class': 15 }
    param['eval_metric'] = 'merror'
    param['min_child_weight'] = 3
    param['nthread'] = 16
    param['colsample_bytree'] = 0.5
    evallist = [(dtrain,'train')]
    bst = xgb.train( param, dtrain, 500, evallist )

    dout = xgb.DMatrix( X_test )
    t_probs = bst.predict(dout)

    t_pred = [prob.tolist().index(max(prob)) for prob in t_probs]

    # compute and plot confusion matrix
    confmat = confusion_matrix(t_test + 1, np.array(t_pred) + 1)
    cm_normalized = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)
    print('Confusion matrix, normalized')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized)
    # plt.show()
    plt.savefig("confmatrix_withdll.png")

    count = 0
    correct = 0
    for true, pred in zip(t_test, t_pred):
        count += 1
        if true == pred:
            correct += 1

    print "Percentage:"
    print float(correct)/count * 100

    # save to prediction file!
    util.write_predictions(t_pred, test_ids, "predictions.csv")

if __name__ == "__main__":
    main()
    
