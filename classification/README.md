# Dataset: 
[] Preprocessed Glass Identiﬁcation Data Set
from UCI’s machine learning data repository.     
[] The training/test sets: 
train.txt, test.txt.  
[] Data description: 
https://archive.ics.uci.edu/ml/datasets/Glass+Identification. 

# Implement Naive Bayes 
The inputs of script are training data and unseen data (testing data).   
The script outputs the accuracy on both training and testing data.   
All feature values are continuous real values thus, we use a Gaussian distribution assumption.   

# Implement kNN 
An object is classiﬁed by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small).   
If k = 1, then the object is simply assigned to the class of that single nearest neighbor.   
The inputs are training data and unseen data (testing data).   
The script outputs the accuracy on both training and testing data.   
Accuracy results when k = 1,3,5,7 and (L1,L2) distances, respectively.   
When computing the training accuracy of kNN, we use leave-one-out strategy, i.e. classifying each training point using the remaining training points.
