# Swingy Monkey Reinforcement Learning

## Dependencies

Make sure you have PyGame installed properly; this can be done through 
Anaconda.

For the SVM, NumPy and sklearn will be necessary. These will run a kind of
supervised learning and thus will depend on these standard libraries.

## How to Run

To run the "Fixed Bins" approach, simply run the `q_fixed_bins.py` file.
This model uses only 24 states of the world to learn, and it does so fairly
quickly.

To run the "25 Bins" approach, simply run `q_few_bins.py`. This will take a lot
longer to produce reasonable results.

To run the "SVM" approach, simply run `svm.py`. This assumes that there is a training
file in place. One has already been provided here. To create your own training data so
that you can see how well SVM would do with your own performance, you can run
`SwingyMonkeyGetTraining.py`. Then, simply run `svm.py`. You may need to delete `training3d.txt`
in the process (by default, this appends).
