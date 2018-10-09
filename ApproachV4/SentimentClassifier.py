import os

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from text_classifier.mnb import MNB

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})


def train_classifier(training, test):
    classifier = NaiveBayesClassifier.train(training)
    print('Classifier Accuracy => ', accuracy(classifier, test))


def classify():

    pos_tweet_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV44/datasets/pos_tweets.txt'
    neg_tweet_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV44/datasets/neg_tweets.txt'

    # Generate Training and Test dataset from the txt files we have
    pos = []
    with open(pos_tweet_path) as f:
        for i in f:
            pos.append([format_sentence(i), 'pos'])

    neg = []
    with open(neg_tweet_path) as f:
        for i in f:
            neg.append([format_sentence(i), 'neg'])

    # next, split labeled data into the training and test data
    training = pos[:int((.8) * len(pos))] + neg[:int((.8) * len(neg))]
    test = pos[int((.8) * len(pos)):] + neg[int((.8) * len(neg)):]

    classifier = MNB()
    file_name = './mnb_trained_classifier.p'
    if os.path.isfile(file_name):
        classifier.import_from_file(file_name)
        print("\nClassifier loaded")
    else:
        # Train the classifier
        # Extract the features and class label from the raw data

        # Split data into training and test data
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        print("\nTraining Random Forest classifier...")
        rfc.learn(x_train, y_train, 22)

        # Save the classifier to disk
        rfc.export(file_name)

        predicted_class_labels = rfc.predict(x_test)

        # Calculate the accuracy of the classifier
        accuracy = rfc.get_classifier_accuracy(predicted_class_labels, y_test)
        print("Classifier accuracy: ", accuracy)