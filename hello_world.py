from sklearn import tree

# Compare apples and oranges. 
# 1. Collect training data
# features = [
#     [140,"smooth"],
#     [130,"smooth"],
#     [150,"bumpy"],
#     [170,"bumpy"]
# ]

# Scikit uses real-valued features (can't use strings as features)
# Weight and texture. "0" is bumpy. "1" is smooth
# Weight and texture. "0" is red. "1" is orange
features = [
    [140,1],
    [130,1],
    [150,0],
    [170,0]
]

#  "0" is apple. "1" is orange
labels = ["apple", "apple", "orange", "orange"]

# Create the classifier (empty box of rules)
classifier = tree.DecisionTreeClassifier()

# 2. Train Classifier. fit() is the training algorithm. 
classifier = classifier.fit(features, labels)
# fit() can be seen as a synomym for "find patterns in training data"

# 3. Make Predictions
print(classifier.predict([[170,0]])) #2 dimensional array needed for python3

# THE MORE SCI-KIT WAY
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0, 0, 1, 1]
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features,labels)
print(classifier.predict([[150,0]])) #2 dimensional array needed for python3