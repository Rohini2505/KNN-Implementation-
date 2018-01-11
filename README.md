# KNN-Implementation-
Implementation of the K-Nearest Neighbors classifier.

Supervised machine learning algorithm, KNN (K-nearest neighbour) Implementation

The implementation should accept two data files as input (both are posted with the assignment): a spam_train.csv file and a spam_test.csv file. Both files contain examples of e-mail messages, with each example having a class label of either "1" (spam) or "0" (no-spam). Each example has 57 (numeric) features that characterize the message. Our classifier should examine each example in the spam test set and classify it as one of the two classes. The classification will be based on an unweighted vote of its k nearest examples in the spam_train set. We will measure all distances using regular Euclidean distance. 

(a) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 without nor-malizing the features.
(b) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 with z-score normalization applied to the features.
