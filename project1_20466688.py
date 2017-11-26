import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


def cal(model, model_name):
    clf = model
    # import data
    data_train = pd.read_csv(r'/Users/mengqian/Desktop/DLproject1/traindata.csv')
    label_train = pd.read_csv(r'/Users/mengqian/Desktop/DLproject1/trainlabel.csv')
    label_test = pd.read_csv(r'/Users/mengqian/Desktop/DLproject1/testdata.csv')

    # data processing
    train_scaled = preprocessing.scale(data_train)
    test_scaled = preprocessing.scale(label_test)
    train_scaled = pd.DataFrame(data=train_scaled)
    label_train = np.ravel(label_train)
    clf.fit(train_scaled, label_train)

    # use Cross Validation to estimate the accuracy of the model
    scores = cross_val_score(clf, train_scaled, label_train, cv=5)
    print(scores)

    # print the result and save as a csv file
    result = clf.predict(test_scaled)
    res = map(lambda x: int(x), result)
    file_name = "/Users/mengqian/Desktop/DLproject1/test_label_%s.csv" % model_name
    np.savetxt(file_name , res, delimiter=",")
    print(res)

models = [
    # Naive Bayes classifier for multivariate Bernoulli models
    (BernoulliNB(), "Naive_Bayes"),

    # Classifier implementing the k-nearest neighbors vote
    (KNeighborsClassifier(algorithm='kd_tree'), "k_nearest"),

    # C-Support Vector Classification
    (SVC(probability=True, kernel='rbf'), "svc")
]

for model, name in models:
    cal(model=model, model_name=name)


