# Median: per demographic (use per-user if no demographic)
# In particular, this is median plays per demographic
# Basically, instead of global_median, you use demographic_median among the user_plays.

import csv
import gzip
import numpy as np

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = '../data/train.csv.gz'
test_file  = '../data/test.csv.gz'
profiles_file = '../data/profiles.csv.gz'
soln_file  = 'demographic_median.csv'

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

# Compute the global median and per-user median.
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

# Compute demographic medians
# Profiles have {user,sex,age,country}
profiles_data = {}
demographic_totals = {}
with gzip.open(profiles_file) as profiles_fh:
    profiles_csv = csv.reader(profiles_fh, delimiter=',', quotechar='"')
    next(profiles_csv, None)
    for row in profiles_csv:
        user = row[0]
        sex = row[1]
        age = row[2]
        country = row[3]

        # Only consider users about which we have data.
        if not user in user_medians:
            continue

        if not user in profiles_data:
            profiles_data[user] = {}

        profiles_data[user]['sex'] = sex
        profiles_data[user]['age'] = age
        profiles_data[user]['country'] = country

        profile_reprs = ['',
                        profiles_data[user]['country'],
                        profiles_data[user]['age'],
                        profiles_data[user]['sex'],
                        profiles_data[user]['sex'] + profiles_data[user]['age'],
                        profiles_data[user]['sex'] + profiles_data[user]['country'],
                        profiles_data[user]['age'] + profiles_data[user]['country'],
                        profiles_data[user]['sex'] + profiles_data[user]['age'] + profiles_data[user]['country']]
        # get unique reprs
        profile_reprs = list(set(profile_reprs))
        for profile_repr in profile_reprs:
            if not profile_repr in demographic_totals:
                demographic_totals[profile_repr] = []

            demographic_totals[profile_repr].append(user_medians[user])

demographic_medians = {}
for demographic, demographic_data in demographic_totals.iteritems():
    demographic_medians[demographic] = np.median(np.array(demographic_data))

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

            result = 0

            if user in profiles_data:
                profile_repr = profiles_data[user]['sex'] + profiles_data[user]['age'] + profiles_data[user]['country']
                result = demographic_totals[profile_repr]
            elif user in user_medians:
                result = user_medians[user]
            else:
                result = global_median

            soln_csv.writerow([id, result])
