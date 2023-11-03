#-------------------------------------------------------------------------
# AUTHOR: Palmer Du
# FILENAME: DuPalmer_svm
# SPECIFICATION: This program trains a Support Vector Machine (svm) using the file optdigits.tra to recognize handwritten digits. The model is then trained
#                with optdigits.tes.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape

highest_accuracy = 0.0
best_parameters = {}

for x in c: 
    for y in degree:
        for z in kernel:
           for a in decision_function_shape:
               
                clf = svm.SVC(C=x, degree=y, kernel=z, decision_function_shape=a)

                clf.fit(X_training, y_training)

                predictions = clf.predict(X_test)

                accuracy = np.mean(predictions == y_test)

                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_parameters = {'C': x, 'degree': y, 'kernel': z, 'decision_function_shape': a}

                    print("Highest SVM accuracy so far:", highest_accuracy)
                    print("Parameters: C={}, degree={}, kernel={}, decision_function_shape={}".format(
                        best_parameters['C'], best_parameters['degree'], best_parameters['kernel'], best_parameters['decision_function_shape']))
                    print()
print("Testing complete!")
