# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
import re
import hashlib
import cPickle as pickle
from collections import Counter
from random import random
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score

import operator
import util

counter = 0
TRAIN_DIR = "../../data/train/"
TEST_DIR = "../../data/test/"

call_set = set([])
global_attributes = {}
# good_keys = pickle.load(open("global_attributes.list", "rb"))

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
        else:
            classes.append(util.malware_classes.index(clazz))

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree, datafile)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def call_feats(tree, datafile):
    global global_attributes
    global counter
    
    counter += 1

    if (counter % 100 == 0):
        print counter

    good_calls = util.all_syscalls
    good_keys = ['filename', 'srcfile', 'protect', 'key', 'value', 'hostname', 'flags']

    call_counter = {}
    for el in tree.iter():
        call = el.tag

        for key, value in el.attrib.iteritems():
            if key in good_keys:
                encoded = key + value
                # if encoded in good_keys:
                if encoded not in global_attributes:
                    global_attributes[encoded] = 0
                else:
                    global_attributes[encoded] += 1

        # attribs = el.attrib

        # if 'hostname' in attribs:
        #     if re.findall(r"duniasex", attribs['hostname']):
        #         if 'duniasex' not in call_counter:
        #             call_counter['duniasex'] = 0
        #         else:
        #             call_counter['duniasex'] += 10
        #             print 'got dunia with {}'.format(datafile)

        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

    good_calls = good_calls + good_keys
    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]

    return np.array(call_feat_array, dtype=int)

## Feature extraction
def main():
    # print "fetching stuff"
    # X_train, t_train, train_ids = create_data_matrix(0, 4000, TRAIN_DIR)
    # print "sorted, pickling"
    # sorted_x = sorted(global_attributes.items(), key=operator.itemgetter(1), reverse=False)

    loaded = pickle.load(open("global_attributes.list", "rb"))
    print loaded[500:]
    # pickle.dump(X_train, open("X_train.nparray", "wb"))
    # pickle.dump(t_train, open("t_train.nparray", "wb"))
    # pickle.dump(train_ids, open("train_ids.nparray", "wb"))
    exit()
    # with open("attribs.list", 'w') as f:
    # exit()
    # X_valid, t_valid, valid_ids = create_data_matrix(2000, 4000, TRAIN_DIR)

    # none_index = util.malware_classes.index("None")
    # t_train_bin = [0 if x == none_index else 1 for x in t_train]
    # t_valid_bin = [0 if x == none_index else 1 for x in t_valid]

    # SVM = SVC(class_weight = 'balanced')
    # SVM.fit(X_train, t_train_bin)
    # pred = SVM.predict(X_valid)

    # count_bin = 0
    # correct_bin = 0
    # for predicted, true in zip(pred, t_valid_bin):
    #     if true == predicted:
    #         correct_bin += 1
    #     count_bin += 1

    # print "REAL partial:"
    # print float(correct_bin) / count_bin * 100

    # t_train_multi = []
    # t_train_ids = []

    # final_prediction = []
    # final_ids = []
    # X_train_multi = []

    # X_valid_multi = []
    # t_valid_multi = []
    # valid_ids_multi = []

    # # find ones to train: the ones that are not None
    # for i, true in enumerate(t_train):
    #     if true != util.malware_classes.index("None"):
    #         X_train_multi.append(X_train[i])
    #         t_train_multi.append(true)

    count = 0
    correct = 0
    # for predicted, true, ID, features in zip(pred, t_valid, valid_ids, X_valid):
    #     if predicted == 0:
    #         final_ids.append(ID)
    #         final_prediction.append(util.malware_classes.index("None"))
    #         count += 1
    #         if true == util.malware_classes.index("None"):
    #             correct += 1
    #     else:
    #         X_valid_multi.append(features)
    #         t_valid_multi.append(true)
    #         valid_ids_multi.append(ID)

    # X_train_multi = np.array(X_train_multi)
    # t_train_multi = np.array(t_train_multi)

    # print "Partial percentage:"
    # print float(correct)/count * 100

    RFC = RandomForestClassifier(n_estimators = 64, n_jobs = -1)
    cv = ShuffleSplit(n = X_train.shape[0], n_iter = 10, test_size=0.2)
    scores = cross_val_score(RFC, X_train, y=t_train, cv=cv, n_jobs = -1)
    print scores

    # for predicted, ID in zip(pred, valid_ids_multi):
    #     final_prediction.append(predicted)
    #     final_ids.append(ID)

    # print final_prediction

    # add # of correct stuff in Malwares
    # for true, predicted in zip(t_valid, pred):
    #     if true == predicted:
    #         correct += 1
    #     else:
    #         print "right was {}, but we said {}".format(true, predicted)
    #     count += 1

    # print "Percentage correct:"
    # print float(correct) / count * 100

    # util.write_predictions(final_prediction, final_ids, "predictions.csv")

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()
    