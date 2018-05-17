# PerceptronSentimentAnalyzer
### Prediction of True/Fake and Pos/Neg labels by building two binary Naive Bayes classifiers on the training data

### Data

The uncompressed archive has the following files:

One file train-labeled.txt containing labeled training data with a single training instance (hotel review) per line (total 960 lines). 

The first 3 tokens in each line are:

a unique 7-character alphanumeric identifier

a label True or Fake

a label Pos or Neg


These are followed by the text of the review.

One file dev-text.txt with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).

One file dev-key.txt with the corresponding labels for the development data, to serve as an answer key.


### Programs


perceplearn.py will learn perceptron models (vanilla and averaged) from the training data, and percepclassify.py will use the models to classify new data. The learning program will be invoked in the following way:

> python perceplearn.py /path/to/input

The argument is a single file containing the training data; the program will learn perceptron models, and write the model parameters to two files: vanillamodel.txt for the vanilla perceptron, and averagedmodel.txt for the averaged perceptron. .
The classification program will be invoked in the following way:

> python percepclassify.py /path/to/model /path/to/input

The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to a file containing the test data file; the program will read the parameters of a perceptron model from the model file, classify each entry in the test data, and write the results to a text file called percepoutput.txt in the same format as the answer key.



