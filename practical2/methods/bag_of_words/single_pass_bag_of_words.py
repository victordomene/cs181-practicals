import os
import numpy as np
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

malware_classes = ["Agent","AutoRun","FraudLoad","FraudPack",
"Hupigon","Krap","Lipler","Magania","None",
 "Poison","Swizzor","Tdss","VB","Virut","Zbot"] 

train_files = [f for f in os.listdir("train") if isfile(join("train", f))]
## interesting files are in onlyfiles[1:]

print "Getting X_train for you, bro. Hold on...\n"
vectorizer = TfidfVectorizer(encoding='latin1')
X_all = vectorizer.fit_transform(open(os.path.join("train", f)).read() for f in train_files[1:])
X_train = X_all[:1999]

print "Got X_train. Happy?\n"

Y_train = []
for f in train_files[1:2000]:
	file_malware = f.split(".", 2)[1]
	Y_train.append(malware_classes.index(file_malware))

np.asarray(Y_train)

X_test = X_all[2000:]

## JOSH CHANGE HERE

# clf = SGDClassifier(alpha=0.001, n_iter=100).fit(X_train, Y_train)
RFC = RandomForestClassifier(n_estimators = 128, n_jobs = -1)
RFC.fit(X_train, Y_train)

# STOP CHANGING BABE

# Y_test = clf.predict(X_test)
Y_test = RFC.predict(X_test)

Y_true = []
for f in train_files[2000:]:
	file_malware = f.split(".", 2)[1]
	Y_true.append(malware_classes.index(file_malware))

count = 0
correct = 0
for pred, true in zip(Y_test, Y_true):
	if (pred == true):
		correct += 1
	count += 1

print "Percentage correct:"
print float(correct)/count * 100
