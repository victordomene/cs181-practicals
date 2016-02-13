# CS181 - Machine Learning
# Author: Victor Domene
# February 10th, 2016

"""
This does regression of various types on the features built and data given.
"""

import csv
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

def RMSE(prediction, target):
	return np.sqrt(np.mean((target - prediction) ** 2))

def stupidcrossval(est, feat, out):
	# cross validate and treat output, it comes out as the negative MSE...

	total_feat = feat.shape[1]
	train_in = feat[total_feat*2/3:]
	train_out = out[total_feat*2/3:]
	est.fit(train_in, train_out)

	val_in = feat[:total_feat*2/3]
	val_out = out[:total_feat*2/3]

	print "# Making predictions..."

	prediction = est.predict(val_in)

	return RMSE(prediction, val_out)

def crossval(est, feat, out):
	# cross validate and treat output, it comes out as the negative MSE...
	
	print "# Fitting and making predictions...\n"

	result = np.array(cross_validation.cross_val_score(est, feat, y = out, n_jobs = -1, scoring = 'mean_squared_error'))
	result = -1 * result
	result = np.sqrt(result)
	return result

# uses AdaBoostRegressor
def regressABR(features, gaps, n, alpha):
	print "# Preparing AdaBoostRegressor..."

	ABR = AdaBoostRegressor(n_estimators = n, learning_rate = alpha)

	print "# AdaBoostRegressor ready!\n"
	print "# Starting cross validation..."

	result = crossval(ABR, features, gaps) 

	print "# The cross validation for n_estimators = {} and alpha = {} gave the following results:".format(n, alpha)
	print(result)
	
	return result

# uses RandomForestRegressor
def regressRF(features, gaps, n):
	print "# Preparing RandomForestRegressor..."

	RF = RandomForestRegressor(n_estimators = n)

	print "# RandomForestRegressor ready!\n"
	print "# Starting cross validation..."

	result = crossval(RF, features, gaps)
	print "# The cross validation for n_estimators = {} gave the following results:".format(n)
	
	print(result)
	return result

# uses Lasso regression
def regressLasso(features, gaps, L1):	
	print "# Preparing Lasso..."

	LAS = Lasso(alpha = L1, fit_intercept = True)

	print "# Lasso ready!"
	print "# Starting cross validation..."

	result = stupidcrossval(LAS, features, gaps)
	print "# The cross validation with alpha = {} gave the following results:".format(L1)
	
	print(result)
	return result

# uses Ridge regression
def regressRidge(featurs, gaps):
	print "# Preparing Ridge..."

	RDG = Ridge()

	print "# Ridge ready!"
	print "# Starting cross validation..."

	result = stupidcrossval(RDG, features, gaps)
	print "# The cross validation gave the following results:"
	
	print(result)
	return result

"""
def regressGP():
	print "# Preparing AdaBoostRegressor..."

	ABR = AdaBoostRegressor(n_estimators = n, learning_rate = alpha)

	print "# AdaBoostRegressor ready!"
	print "# Starting 3-fold cross validation..."

	result = cross_validation.cross_val_score(ABR, features, y = gaps, n_jobs = 1)

	print "# The cross validation scores were:"
	print(result)
"""

if __name__ == "__main__":
	gaps = []
	features = []

	print "###### 10k Dataset Regressions #######\n"
	print "# Starting regressions...\n"
	
	# parses the gaps from data
	print "# Parsing output training data..."

	with open('../data/50000/50000-train.csv', 'r') as train_set:
		c = csv.reader(train_set, delimiter=',')
		next(c)

		for row in c:
			gaps.append(row[-1])

	# parse the features
	print "# Parsing features..."

	with open('../features/50000/50000-train-morgan-1024.csv', 'r') as features_set:
		c = csv.reader(features_set, delimiter=',')
		
		# skip header line
		next(c)

		for row in c: 
			features.append(row[:-1])

	print "# Pre-processing...\n"

	# converts from strings... this is awful but ok.
	features = np.array(features, dtype = int)
	gaps = np.array(gaps, dtype = float)

	# perform a bunch of different regressions
	regressRF(features, gaps, 16)
	regressRF(features, gaps, 32)
	regressRF(features, gaps, 64)
	# regressRF(features, gaps, 128)
	# regressABR(features, gaps, 16, 0.7)
	# regressABR(features, gaps, 16, 1.0)
	# regressABR(features, gaps, 16, 1.5)
	# regressABR(features, gaps, 32, 0.7)
	# regressABR(features, gaps, 32, 1.5)
	# regressABR(features, gaps, 128, 0.7)
	# regressABR(features, gaps, 128, 1.0)
	# regressABR(features, gaps, 128, 1.5)
	# regressLasso(features, gaps, 0.5)
	# regressLasso(features, gaps, 1.0)
	# regressLasso(features, gaps, 1.5)
	regressRidge(features, gaps)
	# regressGP()

	print "###### Done! ######"
