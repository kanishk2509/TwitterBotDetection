from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

"""
The larger the number of trees, the better the performance of the random forest classifier at the expense
of an increased computational cost.
"""


class Classifier:

    def __init__(self):
        self.forest = None

    def learn(self, x_train, y_train, n_trees):
        """
        create and train the random forest classifier
        :param x_train: the input training data
        :param y_train: the input class labels
        :param n_trees: the number of decision trees to use
        :return: n/a
        """
        self.forest = RandomForestClassifier(criterion='entropy', n_estimators=n_trees, random_state=42, n_jobs=2,
                                             max_features=None)
        self.forest.fit(x_train.values.reshape(-1, 1), y_train)

    def predict(self, x):
        """
        wrapper for the RandomForestClassifier predict method
        :param x: the input data for the classifier
        :return: array of class labels
        """

        if self.forest:
            return self.forest.predict(x)
        else:
            return None

    def export(self, path):
        """
        convert the classifier to byte representation and save it to a file
        :param path:
        :return:
        """
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

        with open(path, 'wb') as file:
            pickle.dump(self.forest, file)

    def import_from_file(self, path):
        """
        read in the previously saved classifier
        :param path: path to file
        :return:
        """
        self.forest = pickle.load(open(path, "rb"))

    def get_classifier_accuracy(self, y_true, y_prediction):
        """
        get the overall accuracy of the learned model
        :param y_true: the ground truth (correct labels)
        :param y_prediction: the predicted labels as returned from the classifier
        :return: overall accuracy normalized as a decimal between 0.0 and 1.0
        """
        return accuracy_score(y_true, y_prediction)
