# Practical 3 - Preferences in Streaming Music

In this practical, we performed machine learning on truncated lists of
user / artist play counts, in order to predict how many times each of
these users listened to every artist in the collection.

## Dependencies

This code depends on the following libraries and tools:

* Pandas (`pip install pandas`)
* IPython Notebook (installed via jupyter)
* Nimfa (`pip install nimfa`)
* numpy (of course)

## Directory Structure

This directory is separated into several intuitively-named folders.

In `data`, you should add the files which are downloaded from the Kaggle
site: `artists.csv.gz`, `profiles.csv.gz`, `test.csv.gz`, and `train.csv.gz`.
Our code takes care of reading from the gzipped form.

In `experiments`, you will find the preliminary data analysis.

In `features`, you will code that focused on feature engineering.

In `matfat`, you will find code that used matrix factorization techniques from popular libraries.

In `medians`, you will find code that calculates based on the demographic median.

In `methods`, you will find code that uses global medians, user medians, manual
work, manual factorization, a Nimfa implementation, and a random forests implementation.

Some of the code will deposit results into `predictions` (not pushed for size limits).

After performing experiments, `processed_data` will have `centered_train.csv` and `total_counts.csv`
which can be used in centered tests.

In `references`, you will find some of the helpful papers that were provided for this project.

In `tools`, you will find some more feature centering code.

In `writeup`, the writeup for this practical is included.

In the home directory, you will find this `README.md` as well as the assignment statement
(`practical3-unsupervised.pdf`).
