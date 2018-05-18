## TF Logistic regression
This model detects features from the images using the SIFT algorithm. Then performs a vectorization of this features using a k means algorithm to reduce dimensionality and set the number of features equal for all the features. After this, the model receives a normalized histogram multiplied by the inverse document frequency of each feature. The model is a one layer neural net with K number of features (from k means) and 43 output units (# classes) and a softmax activation function.