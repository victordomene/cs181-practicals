import os
import cPickle as pickle
import random
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import util

def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x

counter = 0
TRAIN_DIR = "../data/train/"
TEST_DIR = "../data/test/"

train_shuffle_directory = shuffle(os.listdir(TRAIN_DIR))
test_shuffle_directory = shuffle(os.listdir(TEST_DIR))

call_set = set([])

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

    # uses likely_syscalls here
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

def main():
    X_train, t_train, train_ids = create_data_matrix(0, 4000, TRAIN_DIR)
    pickle.dump((X_train, t_train, train_ids), open("likely_tags/train.pickle", "wb"))

    X_test, t_test, test_ids = create_data_matrix(0, 4000, TEST_DIR)
    pickle.dump((X_test, t_test, test_ids), open("likely_tags/test.pickle", "wb"))

if __name__ == "__main__":
    main()
