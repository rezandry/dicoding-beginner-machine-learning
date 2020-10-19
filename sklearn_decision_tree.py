# Convert dataset to dataframe
import pandas as pd

# Read dataset
iris = pd.read_csv('Iris.csv')
print(iris.head())

# Remove Property ID
# labels = name of columns
# axis = 1 for columns, 0 for index
# inplace = if False, will make copy of data and make modification, if True, will do modification original data and return None
iris.drop(labels='Id', axis=1, inplace=True)
print(iris.head())

# Separate data from atribut and label
x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']
print(x)
print(y)

from sklearn.tree import DecisionTreeClassifier

# Create model decision tree
tree_model = DecisionTreeClassifier()

# Training model to data
tree_model.fit(x,y)

# Predict model to sample train data
# 77 -> 6.8,2.8,4.8,1.4 -> Iris-versicolor
print(tree_model.predict([[6.8, 2.8, 4.8, 1.4]]))

# Create visualization and create output dot file
from sklearn.tree import export_graphviz
export_graphviz(
    tree_model,
    out_file = "iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True
)
# Open dot file with this terminal command
# xdot iris_tree.dot

