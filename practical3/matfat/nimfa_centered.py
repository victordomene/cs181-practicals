# Nimfa on centered data

import csv
import gzip
import numpy as np
import nimfa

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = '../processed_data/centered_train.csv'
test_file  = '../data/test.csv.gz'
total_counts_file = '../processed_data/total_counts.csv'
soln_file  = 'nimfa_centered.csv'

# Load the training data.
train_data = {}
artists_set = set()
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]

        if not user in train_data:
            train_data[user] = {}

        artists_set.add(artist)
        train_data[user][artist] = float(plays)

# Get all artists
artists_map = {}
for idx, artist in enumerate(artists_set):
  artists_map[artist] = idx

users_map = {}
for idx, user_dict in enumerate(train_data.iteritems()):
  user, _ = user_dict
  users_map[user] = idx

X = np.array([[user_data.get(artist,0) for artist in artists_set] for _, user_data in train_data.iteritems()])

snmf = nimfa.Snmf(X, seed="random_vcol", rank=30, max_iter=12, track_error=True, version='l', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)

print("Nimfa snmf : %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
fit = snmf()
X_fitted = fit.fitted()

# Load User counts
total_counts = {}
with open('total_counts_file', 'r') as total_counts_fh:
  total_counts_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
  next(total_counts_csv)
  for row in total_counts_csv:
    user = row[0]
    total_count = row[1]

    total_counts[user] = int(total_count)

total_counts_median = np.median(total_counts.values())

# Write out test solutions.
with gzip.open(test_file) as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)

        soln_csv.writerow(['Id', 'plays'])
        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            user_idx = users_map[user]
            artist_idx = artists_map[artist]

            multiplier = total_counts.get(user, total_counts_median)
            result = X_fitted[user_idx][artist_idx] * multiplier

            soln_csv.writerow([id, result])
