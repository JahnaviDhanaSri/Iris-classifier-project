import csv
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score

filename = "iris_data.csv"

rows = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
        
X = []
y = []
dictionary = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
for row in rows:
    n=len(row)
    if(n==0):
        break
    x=[]
    for i in range(n-1):
        x.append(float(row[i]))
    X.append(x)
    y.append(dictionary[row[n-1]])

split_ratio = float(input("Give the split ratio between 0 and 1\n"))
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=split_ratio,random_state=3)

# d=DecisionTreeClassifier()
# d=KNeighborsClassifier()
# d=RandomForestClassifier()
# d = SVC()
# d = GaussianProcessClassifier()
# d = MLPClassifier()
# d=AdaBoostClassifier()
# d=GaussianNB()
d=QuadraticDiscriminantAnalysis()

d.fit(x_train,y_train)
y_pred = d.predict(x_test)

print("The accuracy is: ",accuracy_score(y_pred,y_test)*100)    
       
