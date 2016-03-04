# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

import util

TRAIN_DIR = "../../data/train/"

call_set = set([])
global_call = []

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
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
        elif clazz == "None":
            # no malware is 0
            classes.append(0)
        else:
            classes.append(1)

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
    good_calls = util.likely_syscalls

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
    X_train, t_train, train_ids = create_data_matrix(0, 2000, TRAIN_DIR)
    X_valid, t_valid, valid_ids = create_data_matrix(2000, 3807, TRAIN_DIR)

    RFC = RandomForestClassifier(n_estimators = 64, n_jobs = -1)
    RFC.fit(X_train, t_train)
    pred = RFC.predict(X_valid)

    count = 0
    correct = 0
    for true, predicted in zip(t_valid, pred):
        if true == predicted:
            correct += 1
        count += 1

    print "Percentage correct:"
    print float(correct) / count * 100


    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()
    