#-------------------------------------------------------------------------
# AUTHOR: Palmer Du
# FILENAME: DuPalmer_perceptron.py
# SPECIFICATION: this program tries to determine which parameters result in the most accurate perceptron model.
#                it will attempt to classify handwritten digits using sklearn's Percepton and MLPClassifier libraries.
#                it first trains two models, a single perceptron and a multi-layer perceptron, on 'optdigits.tra'
#                with different parameters, and then tests their accuracy on optdigits.tes.
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

p_max_accuracy = 0
mlp_max_accuracy = 0

for x in n:

    for y in r:

        a = [Perceptron, MLPClassifier]

        for z in a:
            if z == Perceptron:
              clf = Perceptron(eta0=x, shuffle=y, max_iter=1000)
            else:
              clf = MLPClassifier(activation='logistic', learning_rate_init=x, hidden_layer_sizes=(25,), shuffle=y, max_iter=1000) 
            
            clf.fit(X_training, y_training)
               
            clf.predict(X_test)
            
            accuracy = clf.score(X_test, y_test)

            if (accuracy > p_max_accuracy):
               if z == Perceptron:
                  p_max_accuracy = accuracy
                  print("Highest Perceptron accuracy so far: " + str(p_max_accuracy) + ", Parameters: learning rate = " + str(x) + ", shuffle = " + str(y))
               else:
                  p_max_accuracy = accuracy
                  print("Highest Perceptron accuracy so far: " + str(mlp_max_accuracy) + ", Parameters: learning rate = " + str(x) + ", shuffle = " + str(y))

            if (accuracy > mlp_max_accuracy):
               if z == Perceptron:
                  mlp_max_accuracy = accuracy
                  print("Highest MLP accuracy so far: " + str(p_max_accuracy) + ", Parameters: learning rate = " + str(x) + ", shuffle = " + str(y))
               else:
                  mlp_max_accuracy = accuracy
                  print("Highest MLP accuracy so far: " + str(mlp_max_accuracy) + ", Parameters: learning rate = " + str(x) + ", shuffle = " + str(y))











