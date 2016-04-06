import numpy as np
import csv
import gzip
import pickle

from sklearn.ensemble import RandomForestRegressor

NUM_ESTIMATORS = 16

# Predict via the user-specific median.
# If the user has no data, use the global median.
users_file = '../data/profiles.csv.gz'
train_file = '../data/train.csv.gz'
test_file  = '../data/test.csv.gz'
soln_file  = '../predictions/random_forests.csv'

# Load the users data. !# CURRENTLY NOT USED
# users = {}
# with gzip.open(users_file, 'r') as users_fh:
#     users_csv = csv.reader(users_fh, delimiter=',', quotechar="'")
#     next(users_csv, None)

#     for row in users_csv:
#         user = row[0]
#         sex = row[1]
#         age = row[2]
#         country = row[3]

#         users[user] = [sex, age, country]

print "# Loading X_train, artists and users..."

# We keep a dictionary that indexes the weird hash into a simple number.
# This is used to track the column for each artist and the row for each
# user in X_train (see below).
artists = {}
artists_index = 0

users = {}
user_index = 0

# Load the training data.
train_data = {}

with gzip.open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        
        if not artist in artists:
            artists[artist] = artists_index
            artists_index += 1

        if not user in users:
            users[user] = user_index
            user_index += 1

        train_data[user][artist] = int(plays)

# X_train will be converted into a np array, so we can use it in RandomForestRegressor
X_train = []

for user, plays_for_artists in train_data.iteritems():
    plays_for_user = [0 for _ in artists.keys()]

    # Here we use artists[aritst] to index artist (a weird hash) into a specific
    # column in X_train.
    for artist, plays in plays_for_artists.iteritems():
        plays_for_user[artists[artist]] = plays

    X_train.append(plays_for_user)

# Finally make it an nparray.
X_train = np.array(X_train)

# Calculate global median and per-user median.
global_median = np.median(X_train[X_train != 0])

median_for_user = []
for user in xrange(len(users)):
    median = np.median(X_train[user][X_train[user] != 0])
    median_for_user.append(median)

print "# Data loaded."

print "# Initializing and fitting one random forest per artist..."

# We create one RandomForestRegressor for each artist
random_forests = [RandomForestRegressor(n_estimators = NUM_ESTIMATORS) for _ in xrange(len(artists))]

for artist_index, forest in enumerate(random_forests):
    if artist_index % 100 == 0:
        print "Current artist: {}".format(artist_index)

    # Use np indexing to fetch only users who listened to this artist;
    # more than that is useless for this particular tree.

    X_artist = X_train[X_train[:, artist_index] != 0]
    Y_artist = X_artist[:, artist_index]

    # This column is Y_artist, so we remove it from the training step
    np.delete(X_artist, artist_index)

    forest.fit(X_artist, Y_artist)

print "# RandomForestRegressors are all ready."

print "# Starting predictions..."

# Write out test solutions.
with gzip.open(test_file, 'r') as test_fh:
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

            try:
                user = users[row[1]]
            except KeyError:
                # If we do not have information on the user, we must 
                # use the global median... Nothing better.

                prediction = global_median
                soln_csv.writerow([id, prediction])
                continue

            try:
                artist = artists[row[2]]
            except KeyError:
                # If we do have the user but not the artist, we use
                # the per-user median... Nothing better.

                prediction = median_for_user[user]
                soln_csv.writerow([id, prediction])
                continue

            if int(id) % 1000 == 0:
                print "Current prediction: {}".format(id)

            # If we already have the answer, we don't want to try to
            # predict it again!
            prediction = None

            if X_train[user, artist] != 0:
                prediction = X_train[user, artist]
            else:
                forest = random_forests[artist]

                # Be careful to use np.delete on a copy, not on X_train
                # itself! That would be a bad idea
                X_train_user = X_train[user]

                # Remove the "0" we have for the artist (otherwise,
                # we would not be in this if case; it's 0 here)
                np.delete(X_train_user, artist)

                # We have a single sample here, so we have to do a reshape
                # to silence a couple warnings...
                X_train_user.reshape(1, -1)

                # Finally predict, given user's history
                prediction = forest.predict(X_train_user)

            soln_csv.writerow([id, prediction])

print "# All predictions ready! Good luck at Kaggle."