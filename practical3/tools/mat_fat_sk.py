import csv
import numpy as np

with open('../processed_data/centered_train.csv', 'r') as centered_train_fh:
    centered_train = csv.reader(centered_train_fh, delimiter=',', quotechar='"')
    users = {}
    artists = {}
    for row in centered_train:
        user_id = row[0]
        artist_id = row[1]
        count = row[2]
        if not user_id in users:
            users[user_id] = {}
        if not artist_id in artists:
            artists[artist_id] = {}

        users[user_id][artist_id] = float(count)

X = np.array([[users[user_id].get(artist_id, 0.0) for artist_id in artists.keys()] for user_id in users.keys()])

from sklearn.decomposition import NMF
model = NMF(n_components=233286, init='random', random_state=0)
model.fit(X)

with open('../results/MF_SK.csv', 'w') as results:


model.components_
