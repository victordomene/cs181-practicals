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

print "# Pickling data..."

# Pickle features!
with open('X_train_RF.pickle', 'w') as outfile:
    pickle.dump(X_train, outfile)

with open('users_RF.pickle', 'w') as outfile:
    pickle.dump(users, outfile)

with open('artists_RF.pickle', 'w') as outfile:
    pickle.dump(artists, outfile)

print "# Data pickled successfully."