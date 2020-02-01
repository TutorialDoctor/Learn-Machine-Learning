# K nearest neighbor classifier
# https://www.youtube.com/watch?v=AoeEHqVSNOw&t=33s


import math

# Distance between two points (training and testing data)
def euclidean_distance(x,y):
    # distance = math.sqrt( ((a[0]-b[0])**2)+((a[1]-b[1])**2) )
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance

class TTDKNN:
    # Memorize the training data
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            # Find closest training point to test point
            label = self.closest(row)
            predictions.append(label)
        return predictions
    # loop over training points and keep track of closest one so far
    def closest(self, row):
        # test to first traiin. shortes dist found so far
        best_dist = euclidean_distance(row, self.X_train[0])
        # index of closest training point
        best_index = 0
        # iterate over all other training points
        for i in range(1, len(self.X_train)):
            dist = euclidean_distance(row, self.X_train[i])
            # if we find a closer one
            if dist < best_dist:
                #update
                best_dist = dist
                # use indext to return clossest training example
                best_index = i
        return self.y_train[best_index]


if __name__ == "__main__":
    import sys
    #from sklearn.datasets import load_iris
    #Classify apples and oranges by weight and texture
    
    features = [[140,1],[130,1],[150,0],[170,0]] # 0 bumpy, 1 smoith
    #print(features.shape)
    labels = [0,0,1,1] #0 is apple, 1 is orange 
    #print(labels.shape)

    #dataset = load_iris()
    #X, y = dataset.data, dataset.target  # pylint: disable=no-member
    clf = TTDKNN()
    #clf.fit(X, y)
    clf.fit(features,labels)
    print(clf.predict([[150,0]]))
    #print(clf.predict([[0, 0, 5, 1.5]]))