
# Classifying eye state from EEG data
This is a student project for KDDM2 on Graz Technical University.
The dataset used was: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State

The best result i could produce was 74% accuracy using gradient boost, validated with 5-fold cross validation.


## Relevant Files
The results presented in the poster for the lecture where produced by the code in following files:
* KNN.py: the KNN classifier used by the orignal authors of the dataset
* gradient_boost.py: my approach: classification via gradient boost

## Other approaches
Other approaches i tried but abandoned can be found in the following files
* ANN.py: artificial neural network approach, tried MLP and LSTM
* SVM.py: support vector machine approach
* wavlet_trans.py: same gradient boost classifier as above, but fed with wavelet transformed input

