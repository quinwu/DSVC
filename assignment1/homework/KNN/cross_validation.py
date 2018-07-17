import pandas as pd
from envs.python36.Lib.collections import Counter
from sklearn.metrics import accuracy_score

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.txt', header=None, names=names)
df.head()

# print(df.values)#查看数据表的值


import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score

X = np.array(df.ix[:, 0:4])  # 矩阵x
y = np.array(df['class'])  # 目标向量

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# use classifier
from sklearn.neighbors import KNeighborsClassifier

# k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict
pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, pred))

# creating odd list of K for KNN
myList = list(range(1, 50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# painting error and K-value
MSE = [1 - x for x in cv_scores]  # error

optimal_k = neighbors[MSE.index(min(MSE))]  # min error of k
print("The optimal number of neighbors is %d" % optimal_k)






#KNN classifier

def train(X_train, y_train):
    # do nothing
    return


def predict(X_train, y_train, x_test, k):
    # list for distance and targets
    distances = []
    targets = []

    for i in range(len(X_train)):
        # compute distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list
        distances.append([distance, i])

    # sort
    distances = sorted(distances)

    # list for target
    for i in range(k):
        index = distances[i][1]  # 取出下标
        targets.append(y_train[index])  # 获取下标为index的类别

        # return most common target
        return Counter(targets).most_common(1)[0][0]


def KNearestNeignbor(X_train, y_train, X_test, predictions, k):
    if k > len(X_train):
        raise ValueError

    # train on the input data
    train(X_train, y_train)

    # loop
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))


predictions = []
try:
    KNearestNeignbor(X_train, y_train, X_test, predictions, 7)

    # transform the list into array
    predictions = np.asarray(predictions)

    # avaluating accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    print('\nThe accuracy of our classifier is %d%%' % accuracy)

except ValueError:
    print('Can\'t have more neighbors than training samples!!!')
