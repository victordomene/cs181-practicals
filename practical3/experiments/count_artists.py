# Get count of artists data we have for each user.

import csv
import gzip
import numpy as np

import matplotlib.pyplot as plt

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = '../data/train.csv.gz'
test_file  = '../data/test.csv.gz'
profiles_file = '../data/profiles.csv.gz'
results_file  = 'count_artists.csv'

# Load the training data.
train_data = {}
with gzip.open(train_file) as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]

        if not user in train_data:
            train_data[user] = {}

        train_data[user][artist] = int(plays)

results = []
with open(results_file, 'w') as results_fh:
    results_csv = csv.writer(results_fh,
                          delimiter=',',
                          quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
    results_csv.writerow(['user', 'artists_count'])

    for user, user_data in train_data.iteritems():
        count = len(user_data)

        results_csv.writerow([user, count])
        results.append(count)

plt.hist(results)
plt.title("Artists Count per User (Train Data)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
