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
    X_valid, t_valid, valid_ids = create_data_matrix(2000, 4000, TRAIN_DIR)

    # separates the t_train only between 0 and 1, where 0 is None and 1 
    # is any Malware
    none = util.malware_classes.index("None")
    t_train_bin = [0 if x == none else 1 for x in t_train]
    t_valid_bin = [0 if x == none else 1 for x in t_valid]

    # train a Random Forest on the data, using a binary classification only
    # (between Malware and None)
    RFC_bin = RandomForestClassifier(n_estimators = 64, n_jobs = -1)
    RFC_bin.fit(X_train, t_train_bin)

    # predict whether the validation inputs are Malwares or Nones
    pred_bin = RFC_bin.predict(X_valid)

    # fetch all datapoints that we considered as Malwares
    X_valid_malware = []
    t_valid_malware = []
    valid_ids_malware = []

    for predicted, ID, true, features in zip(pred_bin, valid_ids, t_valid, X_valid):
        # if we predicted None, this goes to our final prediction
        # otherwise, we add it to X_valid_malware
        if predicted == 0:
            final_prediction.append(none)
            final_ids.append(ID)
        else:
            X_valid_malware.append(features)
            t_valid_malware.append(true)
            valid_ids_malware.append(ID)

    # fetch all the Malwares
    X_train_malware = []
    t_train_malware = []

    for true, features in zip(t_train, X_train):
        if true != util.malware_classes.index("None"):
            X_train_malware.append(features)
            t_train_malware.append(true)

    np.asarray(X_train_malware)
    np.asarray(t_train_malware)

    # train a Random Forest on the data, using now only the Malwares
    RFC_malware = RandomForestClassifier(n_estimators = 64, n_jobs = -1, class_weight = 'balanced')
    RFC_malware.fit(X_train_malware, t_train_malware)
    pred_malware = RFC_malware.predict(X_valid_malware)

    for predicted, ID in zip(pred_malware, valid_ids_malware):
        final_prediction.append(predicted)
        final_ids.append(ID)

    y_pred = [x for (y,x) in sorted(zip(final_ids, final_prediction))]
    y_true = [x for (y,x) in sorted(zip(valid_ids, t_valid))]

    # count = 0
    # correct = 0
    # for pred, true in zip(y_pred, y_true):
    #     if pred == true:
    #         correct += 1
    #     count += 1

    # print "Percentage:"
    # print float(correct)/count * 100

    # add # of correct stuff in Malwares
    confmat = confusion_matrix(y_true, y_pred)
    print np.sum(confmat)

    # compute and plot confusion matrix
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(confmat)
    plt.figure()
    plot_confusion_matrix(confmat)
    # plt.show()
    plt.savefig("confmatrix.png")

    # save to prediction file!
    # util.write_predictions(final_prediction, final_ids, "predictions.csv")

if __name__ == "__main__":
    main()
    
