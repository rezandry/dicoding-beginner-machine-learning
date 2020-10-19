import sklearn
from sklearn import datasets

iris = datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import cross_val_score
from sklearn import tree

clf = tree.DecisionTreeClassifier()

scores = cross_val_score(clf, x, y, cv=5)