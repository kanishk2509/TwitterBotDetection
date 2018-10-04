from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
import re
import string
import time


table = str.maketrans({key: None for key in string.punctuation})


def save(vectorizer, classifier):
    '''
    save classifier to disk
    '''
    with open('model.pkl', 'wb') as file:
        pickle.dump((vectorizer, classifier), file)


def load():
    '''
    load classifier from disk
    '''
    with open('model.pkl', 'rb') as file:
        vectorizer, classifier = pickle.load(file)
    return vectorizer, classifier


def load_data():
    dtype = {"tweets": str, "category": int}
    df = pd.read_csv('cleaned_tweets.csv')

    features = list(df.get('tweets'))
    labels = list(df.get('category'))

    return features, labels


def remove_url(tweet):
    """
    Regex based URL removed. Removes all nonwhitespace characters after http until a whitespace is reached
    :param tweet: Tweet to be checked
    :return: Tweet that is substituted with URL in the place of the actual URL
    """
    return re.sub(r"http\S+", "URL", tweet)


def preprocess(tweet):
    """
    Substitures urls with the string URL. Removes leading and trailing whitespaces
    Removes non latin characters
    :param tweet:
    :return:
    """
    # remove URL
    line = remove_url(str(tweet.strip()))
    # remove non Latin characters
    stripped_text = ''
    for c in line:
        stripped_text += c if len(c.encode(encoding='utf_8')) == 1 else ''

    return stripped_text.translate(table).strip()


def main():

    # load features and labels
    print('Loading data')
    start = time.time()
    features, labels = load_data()
    end = time.time()
    print('CSV Loading time:{diff}'.format(diff=end - start))

    # split data into training / test sets
    print('Splitting data')
    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=0.2,   # use 10% for testing
        random_state=42)

    print("no. of train features: {}".format(len(features_train)))
    print("no. of train labels: {}".format(len(labels_train)))
    print("no. of test features: {}".format(len(features_test)))
    print("no. of test labels: {}".format(len(labels_test)))

    # vectorize email text into tfidf matrix
    # TfidfVectorizer converts collection of raw documents to a matrix of TF-IDF features.
    # It's equivalent to CountVectorizer followed by TfidfTransformer.
    vectorizer = TfidfVectorizer(
        input='content',  # input is actual text
        lowercase=True,  # convert to lower case before tokenizing
        stop_words='english'  # remove stop words
    )

    print('Transforming features')
    start = time.time()
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)
    end = time.time()
    print('Transforming time:{diff}'.format(diff=end - start))

    # train a classifier
    print('Training')
    start = time.time()
    classifier = MultinomialNB()
    classifier.fit(features_train_transformed, labels_train)
    end = time.time()
    print('Training time:{diff}'.format(diff=end - start))

    # save the trained model
    save(vectorizer, classifier)

    # score the classifier accuracy
    print('Scoring')
    print("classifier accuracy {:.2f}%".format(classifier.score(features_test_transformed, labels_test) * 100))
    start = time.time()
    prediction = classifier.predict(features_test_transformed)
    end = time.time()
    print('Testing time:{diff}'.format(diff=end-start))
    fscore = metrics.f1_score(labels_test, prediction, average='macro')
    print("F score {:.2f}".format(fscore))


def test():
    tweet_list = []
    preprocessed = []
    tweet = "hello twitter facebook congratulations"
    tweet2 ="This is a test"
    tweet_list.append(str(tweet))
    tweet_list.append(str(tweet2))
    vectorizer, classifier = load()
    preprocessed.append(preprocess(tweet))
    preprocessed.append(preprocess(tweet2))
    vectorized_tweet = vectorizer.transform(preprocessed)
    prediction = classifier.predict(vectorized_tweet)
    print(prediction)


if __name__ == '__main__':
    #main()
    test()

