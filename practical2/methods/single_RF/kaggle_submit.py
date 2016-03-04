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

def main():
    X_train, t_train, _ = pickle.load(open("../../features/all_tags/train.pickle"))
    X_test, _, test_ids = pickle.load(open("../../features/all_tags/test.pickle"))

    RFC = RandomForestClassifier(n_estimators = 40, n_jobs = -1)
    RFC.fit(X_train, t_train)
    pred = RFC.predict(X_test)

    util.write_predictions(pred, test_ids, "../../predictions/single_RF_predictions.csv")

if __name__ == "__main__":
    main()
    