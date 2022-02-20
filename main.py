"""
The Library used is sklearn,
"""

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB


def get_Acc(Yprd, y):
    return (Yprd / y) * 100


X, Y = load_iris(return_X_y=True)
Xdescription = load_iris().feature_names
Ydescription = load_iris().target_names
Xtrain, Xtest, YTrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=42)
print((Xtrain, Xtrain.shape))
GNB = GaussianNB()
GNB.fit(Xtrain[:, :2], YTrain)

YPredection = GNB.predict(Xtest[:, :2])

Total_Instances = Xtest.shape[0]
incorrect_Instances = (Ytest != YPredection).sum()
acc = get_Acc(Total_Instances - incorrect_Instances, Total_Instances)

print("N0. of Instances: " + str(Total_Instances))
print("No. of Incorrect Guesses: " + str(incorrect_Instances))
print("Accuracy: " + str(acc) + "%")

# Plotting decision region sepal length [cm]
plot_decision_regions(Xtrain[:, :2], YTrain, clf=GNB, legend=2)

# Adding axes annotations
plt.xlabel('sepal length')
plt.ylabel('petal length [cm]')
plt.title('GNB on Iris')
plt.show()
