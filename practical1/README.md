# Practical 1

This is the GitHub repository for CS181 Practical #1.

## Directory Structure

On this repository, we will not upload heavy files such as .csv.
Instead, only the scripts will be here, and the folders where everything
should be set up.

The names are pretty descriptive: benchmarks will end up in that folder,
while features should go in the features folder.

In methods, we have several files that we used to run our different tests.

In features, we would have the data for features, as well as a lot of code for
building features.

Finally, the writeup is in the writeup folder.

## What do we need to run these?

The features were generated using RDKit, which can be a bit annoying to install.

The machine learning algorithms come mostly from Sklearn. In the case of
neural networks, we used PyBrain. These can be easily installed with Pip, and
they require NumPy.

## Team Members
Victor Domene
Henrique Vaz
Stefan Gramatovici

## Best Score on Kaggle
0.5994, using the Random Forest Regressor with 64 estimators, 1024 features
from Morgan Fingerprints and trained on the entire 1 million molecules dataset.
