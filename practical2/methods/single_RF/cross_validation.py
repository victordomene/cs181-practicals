import os
import cPickle as pickle
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score
import util

N = 40
validation_size = 0.2

def main():
    print "# Loading Features..."
    X_train, t_train, train_ids = pickle.load(open("../../features/all_tags/train.pickle"))

    print "# Training RandomForestClassifier with n_estimators = {}...".format(N)
    RFC = RandomForestClassifier(n_estimators = N, n_jobs = -1)

    print "# Running Cross Validation with test_size = {}...".format(validation_size)
    cv = ShuffleSplit(n = X_train.shape[0], n_iter = 10, test_size=validation_size)
    scores = cross_val_score(RFC, X_train, y=t_train, cv=cv, n_jobs = -1)
    
    print "# Cross Validation Scores"
    print scores

    print "# Average Score"
    print sum(scores)/len(scores)

    print "# Done!"

if __name__ == "__main__":
    main()
    