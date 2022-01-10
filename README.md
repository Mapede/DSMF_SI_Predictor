# DSMF_SI_Predictor
This contains the code needed to run the data-driven speech intelligibility predictor proposed in "Training Data-Driven Speech Intelligibility Predictors on Heterogeneous Listening Test Data".
The network was trained using Tensorflow 2.1 and will not work with Tensorflow 1.

network.py contains the Tensorflow 2 architecture.
functions.py contains functionality for preprocessing and loading network weights.
example.py can be used to run the predictor on a pair of single channel .wav files.
