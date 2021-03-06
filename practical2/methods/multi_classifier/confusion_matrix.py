import os
import cPickle as pickle
from collections import Counter
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

N = 40

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
    print "# Loading features..."
    X_train_all, t_train_all, train_all_ids = pickle.load(open("../../features/all_tags/train.pickle"))

    print "# Separating features into two simple sets..."
    X_train, t_train, train_ids = X_train_all[:2000], t_train_all[:2000], train_all_ids[:2000]
    X_test, t_test, test_ids = X_train_all[2000:], t_train_all[2000:], train_all_ids[:2000]

    # separates the t_train only between 0 and 1, where 0 is None and 1 
    # is any Malware
    none = util.malware_classes.index("None")
    t_train_bin = [0 if x == none else 1 for x in t_train]
    t_test_bin = [0 if x == none else 1 for x in t_test]

    # train a Random Forest on the data, using a binary classification only
    # (between Malware and None)
    print "# Training RandomForestClassifier with n_estimators = {}, for a binary classification between Malware or None...".format(N)
    RFC_bin = RandomForestClassifier(n_estimators = 64, n_jobs = -1)
    RFC_bin.fit(X_train, t_train_bin)

    # predict whether the validation inputs are Malwares or Nones
    print "# Predicting Malware vs None..."
    pred_bin = RFC_bin.predict(X_test)

    # fetch all datapoints that we considered as Malwares
    X_test_malware = []
    t_test_malware = []
    test_ids_malware = []

    for predicted, ID, true, features in zip(pred_bin, test_ids, t_test, X_test):
        # if we predicted None, this goes to our final prediction
        # otherwise, we add it to X_test_malware
        if predicted == 0:
            final_prediction.append(none)
            final_ids.append(ID)
        else:
            X_test_malware.append(features)
            t_test_malware.append(true)
            test_ids_malware.append(ID)

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
    print "# Training another RandomForestClassifier with n_estimators = {}, for a multi-class classification between only Malwares..."
    RFC_malware = RandomForestClassifier(n_estimators = N, n_jobs = -1, class_weight = 'balanced')
    RFC_malware.fit(X_train_malware, t_train_malware)
    
    print "# Predicting whatever we had not classified as None before..."
    pred_malware = RFC_malware.predict(X_test_malware)

    for predicted, ID in zip(pred_malware, test_ids_malware):
        final_prediction.append(predicted)
        final_ids.append(ID)

    y_pred = [x for (y,x) in sorted(zip(final_ids, final_prediction))]
    y_true = [x for (y,x) in sorted(zip(test_ids, t_test))]

    print "# Plotting confusion matrix..."
    # compute and plot confusion matrix
    confmat = confusion_matrix(y_true, y_pred)
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
    for true, pred in zip(y_true, y_pred):
        count += 1
        if true == pred:
            correct += 1

    print "# Percentage of correct classifications:"
    print float(correct)/count * 100

    print "# Done!"

if __name__ == "__main__":
    main()
    
