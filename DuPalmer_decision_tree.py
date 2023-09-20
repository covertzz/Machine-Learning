#-------------------------------------------------------------------------
# AUTHOR: Palmer Du
# FILENAME: DuPalmer_decision_tree
# SPECIFICATION: this program will produce a decision tree based on a variety of factors using the ID3 algorithm. Input is 
#                taken from a file named "contact_lens.csv" which is included in the github repo.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2-3 hours of coding/debugging
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
import numpy
db = []
X = []
Y = []

#reading the data in a csv file
with open("c:\\Users\\Palmer\\Downloads\\contact_lens.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
for i in range(len(db)):
   X.append([0,0,0,0])
   if db[i][0] == "Young":
      X[i][0] = 1
   elif db[i][0] == "Prepresbyopic":
      X[i][0] = 2
   elif db[i][0] == "Presbyopic":
      X[i][0] = 3

   if db[i][1] == "Myope":
      X[i][1] = 1
   elif db[i][1] == "Hypermetrope":
      X[i][1] = 2

   if db[i][2] == "Yes":
      X[i][2] = 1
   elif db[i][2] == "No":
      X[i][2] = 2 

   if db[i][3] == "Reduced":
      X[i][3] = 1
   elif db[i][3] == "Normal":
      X[i][3] = 2
   
print(X)

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for i in range(len(db)):
   if db[i][4] == "Yes":
      Y.append(1)
   else:
      Y.append(2)

print(Y)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show() 
