import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Feature Names
print(iris.feature_names)
print()

print(iris.target_names)
print()

# First flower
print(iris.data[0])
print()

# Label Names. "0" is satosa, 1 is versicolor, 2 is virgnica
print(iris.target[0])
print()

# For all labels in data, print label, label name, flower data
for i in range(len(iris.target)):
    print("Example {}: label:{} features: {} ".format(i,iris.target[i],iris.data[i]))

# Training Data
test_indices = [0,50,100]
# Remove targets
train_target = np.delete(iris.target, test_indices)
# Remove data
train_data = np.delete(iris.data, test_indices, axis=0)

# Test Data
test_target = iris.target[test_indices]
test_data = iris.data[test_indices]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_data, train_target)

# If they match, the model got them all right! Let's see!
print(test_target)
print(classifier.predict(test_data))