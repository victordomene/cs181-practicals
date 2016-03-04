# Practical 2 - Identifying Malware

In this practical, we were meant to be able to identify a set of malwares, given
a dataset of around 3086 malwares. The final test dataset contains 3724 files
to predict.

## Dependencies

This code depends on the following libraries:

* Scikit Learn (`pip install sklearn`)
	* RandomForestClassifier
	* LinearSVM
	* MultinomialNB
	* TfIdfVectorizer
	* cross_validation
* XGBoost (`pip install xgboost`)
* numpy (of course)

## Directory Structure

This directory is separated into several intuitively-named folders. 

In `data`, there should be two folders `train` and `test` that will contain the training
and test data, accordingly (not included for space).

In `predictions`, there should be some predictions that we have run.

In `methods`, there should be all of the methods used. The methods include
a single RandomForest for all the data; two Random Forest for a one-vs-rest
approach for Nones; and XGBoost for all the data. There is also the code for
the `bag_of_words` approach.

In `features`, we can find `all_tags` and `likely_tags`, which were the two
sets of features used (other than `bag_of_words`). Bag of Words' feature
engineering is done inside its own file in `method`, so we do not include
those here.

In `writeup`, the writeup for this practical is included.

