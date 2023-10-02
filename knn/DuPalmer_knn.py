#-------------------------------------------------------------------------
# AUTHOR: Palmer Du 
# FILENAME: DuPalmer_knn.py
# SPECIFICATION: This program takes input from binary_points.csv. It calculates the LOOCV error rate for 1NN.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hrs
#-----------------------------------------------------------*/

from sklearn.neighbors import KNeighborsClassifier
import csv
import copy

db = []
error = 0

with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0:
         db.append (row)

for i in range(len(db)):

    X = copy.deepcopy(db)
    for j in range(len(X)) :
       for k in range(2):
          X[j][k] = float(X[j][k])
    X.pop(i)

    Y=[]
    for x in X:
       if x[2] == "-":
          Y.append(0.0)
          del x[2]
       elif x[2] == "+":
          Y.append(1.0)
          del x[2]
    

    testSample = copy.deepcopy(db[i])
    testSample[0] = float(testSample[0])
    testSample[1] = float(testSample[1])
    if testSample[2] == "-":
       testSample[2] = 0.0
    elif testSample[2] == "+":
       testSample[2] = 1.0

    

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    class_predicted = clf.predict([[testSample[0],testSample[1]]])[0]
   
    if class_predicted != testSample[2]:
       error += 1

print("Error Rate: ")
print((error/len(db)))





