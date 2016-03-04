import os
import numpy as np
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors

# def write_predictions(predictions, ids, outfile):
#     """
#     assumes len(predictions) == len(ids), and that predictions[i] is the
#     index of the predicted class with the malware_classes list above for 
#     the executable corresponding to ids[i].
#     outfile will be overwritten
#     """
#     with open(outfile,"w+") as f:
#         # write header
#         f.write("Id,Prediction\n")
#         for i, history_id in enumerate(ids):
#             f.write("%s,%d\n" % (history_id, predictions[i]))


malware_classes = ["Agent","AutoRun","FraudLoad","FraudPack",
"Hupigon","Krap","Lipler","Magania","None",
 "Poison","Swizzor","Tdss","VB","Virut","Zbot"] 

## FIRST WE SEPARE BETWEEN MALWARE VS NON-MALWARE

## define the bin files: they will contain every single file
train_bin_files = [f for f in os.listdir("train") if isfile(join("train", f))]
test_bin_files = [f for f in os.listdir("test") if isfile(join("test", f))]

## files[0] is a shitty file. Remove it
train_bin_files.pop(0)
test_bin_files.pop(0)

## Get ifidf matrix for all files
print "Getting X_train_bin for you, bro. Hold on...\n"
vectorizer = TfidfVectorizer(encoding='latin1')
X_train_bin = vectorizer.fit_transform(open(os.path.join("train", f)).read() for f in train_bin_files[2100:])
print "Got X_train_bin. Happy?\n"

## define Y_train
Y_train_bin = []
for f in train_bin_files[2100:]:
	file_malware = f.split(".", 2)[1]
	if (file_malware == "None"):
		Y_train_bin.append(0)
	else:
		Y_train_bin.append(1)

## make Y_train_bin a np arrays instead of list
np.asarray(Y_train_bin)

## define ifidf matrix for test set. Use everyone again
print "Getting X_test_bin for you, bro. Hold on...\n"
X_test_bin = vectorizer.transform(open(os.path.join("train", f)).read() for f in train_bin_files[:2100])
print "Got X_test_bin"

## use Random Forest to fit data
print "SVM started working its ass off to help you"
RFC = RandomForestClassifier(n_estimators = 1024, n_jobs = -1)
RFC.fit(X_train_bin, Y_train_bin)
print "Done"

## get predictors
Y_test_bin = RFC.predict(X_test_bin)

print np.count_nonzero(Y_test_bin)

Y_true_bin = []
count_malware = 0
count_none = 0
for f in train_bin_files[:2100]:
	file_malware = f.split(".", 2)[1]
	if (file_malware == "None"):
		Y_true_bin.append(0)
		count_none += 1
	else:
		Y_true_bin.append(1)
		count_malware += 1

#print Y_true_bin

#count = 0
correct_malware = 0
correct_none = 0
for pred, true in zip(Y_test_bin, Y_true_bin):
	if (pred == true):
		if (pred == 0):
			correct_malware += 1
		else:
			correct_none += 1
	#count += 1

#print count

print "Percentage correct malware:"
print float(correct_malware)/count_malware * 100

print "Percentage correct none:"
print float(correct_none)/count_none * 100

print "Final Percentage Correct:"
print float(correct_none + correct_malware)/(count_none + count_malware) * 100


# ## NOW START CLASSIFICATION BASED ON MALWARE ONLY

# ## the malware train set will only contain actual malwares
# train_malware_files = [f for f in train_bin_files[2100:] if f.split(".", 2)[1] != "None"]
# ## the malware test set will only contain those files that were classified as malware
# test_malware_files = [f for f in train_bin_files[:2100] if Y_test_bin[train_bin_files.index(f)] == 1]

# ## Get ifidf matrix for malware files
# print "Getting X_train_malware for you, bro. Hold on...\n"
# vectorizer = TfidfVectorizer(encoding='latin1')
# X_train_malware = vectorizer.fit_transform(open(os.path.join("train", f)).read() for f in train_malware_files)
# print "Got X_train_malware. Happy?\n"

# Y_train_malware = []
# for f in train_malware_files:
# 	file_malware = f.split(".", 2)[1]
# 	Y_train_malware.append(malware_classes.index(file_malware))

# ## make Y_train_malware a np array instead of list
# np.asarray(Y_train_malware)

# ## define ifidf matrix for test set. Use malware only
# print "Getting X_test_malware for you, bro. Hold on...\n"
# X_test_malware = vectorizer.transform(open(os.path.join("train", f)).read() for f in test_malware_files)
# print "Got X_test_malware"

# ## use Random Forest to fit data
# print "RF started working its ass off to help you"
# #RFC = RandomForestClassifier(n_estimators = 256, n_jobs = -1)
# clf = neighbors.KNeighborsClassifier(n_neighbors=45)
# clf.fit(X_train_malware, Y_train_malware)
# print "Done"

# ## get predictors
# Y_test_malware = clf.predict(X_test_malware)

# ## NOW WE CONCATENATE THE TWO RESULTS

# ## The philosphy is that files predicted to be non-malwares remain non-malwares
# ## and malwares follow the malware prediction

# Y_test = []
# malware_count = 0
# for f in train_bin_files[:2100]:
# 	if (Y_test_bin[train_bin_files.index(f)] == 0):
# 		Y_test.append(malware_classes.index("None"))
# 	else:
# 		Y_test.append(Y_test_malware[malware_count])
# 		malware_count += 1

# np.asarray(Y_test)

# # ids_test = []
# # for f in test_files[1:]:
# # 	ID = f.split(".", 1)[0]
# # 	ids_test.append(ID)


# Y_true = []
# for f in train_bin_files[:2100]:
# 	file_malware = f.split(".", 2)[1]
# 	Y_true.append(malware_classes.index(file_malware))

# count = 0
# correct = 0
# for pred, true in zip(Y_test, Y_true):
# 	if (pred == true):
# 		correct += 1
# 	count += 1

# print "Percentage correct:"
# print float(correct)/count * 100

# # # write_predictions(Y_test, ids_test, "bag_predictions_sgd.csv")
