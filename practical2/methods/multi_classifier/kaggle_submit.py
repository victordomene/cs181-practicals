import os
import cPickle as pickle
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import util

N = 40

def main():
    final_ids = []
    final_prediction = []

    # fetch features for training and test data
    # substitute for a pickle load, for training data!
    print "# Loading Features..."
    X_train, t_train, train_ids = pickle.load(open("../../features/all_tags/train.pickle"))
    X_test, t_test, test_ids = pickle.load(open("../../features/all_tags/test.pickle"))

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

    print "# Predicting Malware vs None..."
    # predict whether the testation inputs are Malwares or Nones
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

    print "# Training another RandomForestClassifier with n_estimators = {}, for a multi-class classification between only Malwares..."
    # train a Random Forest on the data, using now only the Malwares
    RFC_malware = RandomForestClassifier(n_estimators = 64, n_jobs = -1, class_weight = 'balanced')
    RFC_malware.fit(X_train_malware, t_train_malware)
    
    print "# Predicting whatever we had not classified as None before..."
    pred_malware = RFC_malware.predict(X_test_malware)

    for predicted, ID in zip(pred_malware, test_ids_malware):
        final_prediction.append(predicted)
        final_ids.append(ID)

    util.write_predictions(final_prediction, final_ids, "predictions.csv")

    print "# Done!"

if __name__ == "__main__":
    main()
    
