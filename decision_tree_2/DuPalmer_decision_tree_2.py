#-------------------------------------------------------------------------
# AUTHOR: Palmer Du 
# FILENAME: DuPalmer_decision_tree_2.py
# SPECIFICATION: This program uses contact_lens_training_1.csv, contact_lens_training_2.csv, contact_lens_training_3.csv, 
#                and contact_lens_test.csv. This program calculated the average performance of a decision tree created using each
#                of the three training sets and a pre-pruning strategy of setting the max depth equal to three. Each training set 
#                is then averaged over 10 trials, which is then outputted. The goal of this program is to determine which training
#                set has the best performance, since each set has a different size.

# FOR: CS 4210- Assignment #2
# TIME SPENT: about 2 hours
#-----------------------------------------------------------*/
from sklearn import tree
import matplotlib.pyplot as plt

import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    for i in range(len(dbTraining)):
            X.append([0,0,0,0])
            
            if dbTraining[i][0] == "Young":
                X[i][0] = 1
            elif dbTraining[i][0] == "Prepresbyopic":
                X[i][0] = 2
            elif dbTraining[i][0] == "Presbyopic":
                X[i][0] = 3

            if dbTraining[i][1] == "Myope":
                X[i][1] = 1
            elif dbTraining[i][1] == "Hypermetrope":
                X[i][1] = 2

            if dbTraining[i][2] == "Yes":
                X[i][2] = 1
            elif dbTraining[i][2] == "No":
                X[i][2] = 2 

            if dbTraining[i][3] == "Reduced":
                X[i][3] = 1
            elif dbTraining[i][3] == "Normal":
                X[i][3] = 2

            if dbTraining[i][4] == "Yes":
                Y.append(1)
            elif dbTraining[i][4] == "No":
                Y.append(2)

    sum = 0.0
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0: #skipping the header
                    dbTest.append (row)

        Z = []
        for i in range(len(dbTest)):
            if dbTest[i][4] == "Yes":
                Z.append(1)
            else:
                Z.append(2)

        accuracy = 0.0
        count2=0
        for j in dbTest:
            prediction = [[0,0,0,0]]
            
            if j[0] == "Young":
                prediction[0][0] = 1
            elif j[0] == "Prepresbyopic":
                prediction[0][0] = 2
            elif j[0] == "Presbyopic":
                prediction[0][0] = 3

            if j[1] == "Myope":
                prediction[0][1] = 1
            elif j[1] == "Hypermetrope":
                prediction[0][1] = 2

            if j[2] == "Yes":
                prediction[0][2] = 1
            elif j[2] == "No":
                prediction[0][2] = 2 

            if j[3] == "Reduced":
                prediction[0][3] = 1
            elif j[3] == "Normal":
                prediction[0][3] = 2
            
            if(clf.predict(prediction)[0] == Z[count2]):
                accuracy += 1
            count2 += 1
        sum += accuracy/len(dbTest)
        
    sum = sum/10
    print("The aveage accuracy for " + ds + " is: " + str(sum))
