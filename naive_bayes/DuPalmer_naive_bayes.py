#-------------------------------------------------------------------------
# AUTHOR: Palmer Du
# FILENAME: DuPalmer_naive_bayes.py
# SPECIFICATION: this program takes training data from weather_training.csv and uses the naive bayes alorithm. Test data comes from
#                 weather_test.csv. This program only outputs guesses for classification confidence values above 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0:
         dbTraining.append(row)

def valueToInt(array, set):
    for i in range(len(set)):
        array.append([0,0,0,0])
        if set[i][1] == "Sunny":
            array[i][0] = 1
        elif set[i][1] == "Overcast":
            array[i][0] = 2
        elif set[i][1] == "Rain":
            array[i][0] = 3

        if set[i][2] == "Hot":
            array[i][1] = 1
        elif set[i][2] == "Mild":
            array[i][1] = 2
        elif set[i][2] == "Cool":
            array[i][1] = 3

        if set[i][3] == "High":
            array[i][2] = 1
        elif set[i][3] == "Normal":
            array[i][2] = 2 

        if set[i][4] == "Strong":
            array[i][3] = 1
        elif set[i][4] == "Weak":
            array[i][3] = 2


X = []
valueToInt(X, dbTraining)

Y = []
for i in range(len(dbTraining)):
    if dbTraining[i][5] == "Yes":
        Y.append(1)
    elif dbTraining[i][5] == "No":
        Y.append(2)

clf = GaussianNB()
clf.fit(X, Y)

dbTest = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
        if i > 0:
            dbTest.append (row)

temp = []
valueToInt(temp,dbTest)
for i in range(len(temp)):    
    prediction = clf.predict_proba([temp[i]])[0]
    if max(prediction) == prediction[0]:
        dbTest[i][5] = "Yes"
    elif max(prediction) == prediction[1]:
        dbTest[i][5] = "No"
    dbTest[i].append(max(prediction))

print("%-7s %-11s %-14s %-11s %-9s %-10s %-10s"% ("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"))
for i in dbTest:
    if i[6] > 0.75:
        print("%-7s %-11s %-14s %-11s %-9s %-10s %-10s"% (i[0], i[1], i[2], i[3], i[4], i[5], i[6]))

