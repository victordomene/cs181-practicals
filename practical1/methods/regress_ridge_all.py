# CS181 - Machine Learning
# Author: Victor Domene
# February 10th, 2016

"""
This does regression of various types on the features built and data given.
"""

import csv
import gzip
import numpy as np
from sklearn.linear_model import Ridge

gaps = []
features = []

print "###### 1M Dataset Regressions #######\n"
print "# Starting regressions...\n"

# parses the gaps from data
print "# Parsing output training data..."

with gzip.open('../data/train.gz', 'r') as train_set:
	c = csv.reader(train_set, delimiter=',')
	next(c)
	
	for row in c:
		gaps.append(row[-1])

# parse the features
print "# Parsing features..."

with open('../features/train-morgan-1024.csv', 'r') as features_set:
	c = csv.reader(features_set, delimiter=',')
	
	# skip header line
	next(c)

	for row in c: 
		features.append(row[:-1])

print "# Pre-processing...\n"

# converts from strings... this is awful but ok.
features = np.array(features, dtype = int)
gaps = np.array(gaps, dtype = float)

print "# Preparing Random Forest..."
RD = Ridge()

print "# Fitting training data..."
RD.fit(features, gaps)

print "# Predicting tests..."

with open('../benchmarks/FINAL-Ridge-morgan-1024.csv', 'w') as output:
	w = csv.writer(output, delimiter = ',')

	w.writerow(['Id', 'Prediction'])

	with open('../features/test-morgan-1024.csv', 'r') as test_set:
		c = csv.reader(test_set, delimiter=',')

		# skip headers
		next(c)
		
		i = 1
		for row in c:
			if (i % 10000 == 0):
				print i

			# skip id and smile
			result = RD.predict(np.array(row[:-2], dtype = int))
			w.writerow([i] + result.tolist())
			i += 1

print "###### Done! ######"
