from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os


class MNB:

    def __init__(self):
        self.mnb = None

    def learn(self, x_train, y_train):
        """
        create and train the random forest classifier
        :param x_train: the input training data
        :param y_train: the input class labels
        :param n_trees: the number of decision trees to use
        :return: n/a
        """
        self.mnb = MultinomialNB(alpha=0.0009)
        self.mnb.fit(x_train, y_train)

    def predict(self, x):
        """
        wrapper for the RandomForestClassifier predict method
        :param x: the input data for the classifier
        :return: array of class labels
        """

        if self.mnb:
            return self.mnb.predict(x)
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
            pickle.dump(self.mnb, file)

    def import_from_file(self, path):
        """
        read in the previously saved classifier
        :param path: path to file
        :return:
        """
        self.mnb = pickle.load(open(path, "rb"))

    def get_classifier_accuracy(self, y_true, y_prediction):
        """
        get the overall accuracy of the learned model
        :param y_true: the ground truth (correct labels)
        :param y_prediction: the predicted labels as returned from the classifier
        :return: overall accuracy normalized as a decimal between 0.0 and 1.0
        """
        return accuracy_score(y_true, y_prediction)
