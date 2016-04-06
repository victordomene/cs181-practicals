import csv
import gzip

train_fh = gzip.open('../data/train.csv.gz')
train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')

next(train_csv)

users = {}
for row in train_csv:
    user_id = row[0]
    artist_id = row[1]
    count = row[2]

    if not user_id in users:
        users[user_id] = {}
    users[user_id][artist_id] = int(count)

# Centering
for user_id in users:
    total_count = sum(users[user_id].values())
    for artist_id in users[user_id]:
        users[user_id][artist_id] /= float(total_count)

with open('../processed_data/centered_train.csv', 'w') as centered_train:
    for user_id in users:
        for artist_id in users[user_id]:
            centered_train.write('{0},{1},{2}\n'.format(user_id, artist_id, users[user_id][artist_id]))
