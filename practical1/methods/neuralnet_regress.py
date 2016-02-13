# CS181 - Machine Learning
# Author: Victor Domene
# February 10th, 2016

"""
This code runs a simple PyBrain neural network on the data, and produces
a csv output on the Kaggle format.
"""

import csv
import gzip
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

gaps = []
features = []

print "# Starting network..."

net = buildNetwork(1024, 3, 1, bias = True)
ds = SupervisedDataSet(1024, 1)

# parses the gaps from data
print "# Parsing data into net's dataset..."

with gzip.open('../data/train.gz', 'r') as train_set:
	with open('../features/train-morgan-1024.csv', 'r') as features_set:
		c_out = csv.reader(train_set, delimiter=',')
		c_in = csv.reader(features_set, delimiter=',')

		# skip headers
		next(c_out)
		next(c_in)

		# adds stuff to datasets appropriately
		for irow, orow in zip(c_in, c_out):
			ds.addSample(tuple(irow[:-1]), orow[-1])

print "# Dataset built!"

print "# Training on dataset..."

trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence(maxEpochs = 60, continueEpochs = 10, validationProportion = 0.33)

print "# Parsing test dataset, and predicting..."

with open('../benchmarks/UntilConvergenceNN-morgan-1024.csv', 'w') as output:
	w = csv.writer(output, delimiter = ',')

	w.writerow(['Id', 'Prediction'])

	with open('../features/test-morgan-1024.csv', 'r') as test_set:
		c = csv.reader(test_set, delimiter=',')

		# skip headers
		next(c)
		
		i = 1
		for row in c:
			# skip id and smile
			result = net.activate(row[:-2])
			w.writerow([i] + result.tolist())
			i += 1

print "# Done!"
