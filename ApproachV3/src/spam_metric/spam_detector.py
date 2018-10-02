import math
import os
import json
import pandas as pd
import functools
import random
from collections import defaultdict
from copy import deepcopy
import re

data_columns = {"id": int, "text": str, "source": str, "user_id": str, "truncated": str, "in_reply_to_status_id": str,
                "in_reply_to_user_id": str, "in_reply_to_screen_name": str, "retweeted_status_id": str, "geo": str,
                "place": str, "contributors": str, "retweet_count": int, "reply_count": str, "favorite_count": str,
                "favorited": str, "retweeted": str, "possibly_sensitive": str, "num_hashtags": int, "num_urls": str,
                "num_mentions": str, "created_at": str, "timestamp": str, "crawled_at": str, "updated": str}

def sum(dict):
    return functools.reduce(lambda x, y: x + y, dict.values())

def remove_url(tweet):
    return re.sub(r"http\S+", "URL", tweet)

# расчитываем условную вероятность заданной feature для заданного класса
def conditionalProbability(model, word, klass):
    (docCount, wordSize, wordProb, D) = model
    return float(wordProb[klass].get(word, 0) + 1) / (sum(wordProb[klass]) + len(D))


def classify(model, js):
    (docCount, wordSize, wordProb, D) = model
    Gp = Bp = 0;
    for w in getFeatures(js):
        Gp += math.log(conditionalProbability(model, w, 'G'), 2)
        Bp += math.log(conditionalProbability(model, w, 'B'), 2)
        #Gp += conditionalProbability(model, w, 'G')
        #Bp += conditionalProbability(model, w, 'B')

    Gp += math.log(float(docCount['G']) / sum(docCount))
    Bp += math.log(float(docCount['B']) / sum(docCount))
    #Gp += float(docCount['G']) / sum(docCount)
    #Bp += float(docCount['B']) / sum(docCount)

    # Convert to a single value between -1 to 1,
    # with -1 being completely confident that it is spam and
    # 1 being completely confident that it is not spam

    percentage = 0

    #return "G" if Gp > Bp else "B"
    if Gp > Bp:
        percentage = Gp / (Gp + Bp)

    elif Bp > Gp:
        percentage = -1 * Bp / (Gp + Bp)

    return percentage

def readModel(fileName):
    fp = open(fileName, 'r')
    datum = []
    for x in range(4):
        datum.append(json.loads(fp.readline().strip()))
    fp.close()
    return datum


def writeModel(fileName, docCount, wordSize, wordProb, D):
    fp = open(fileName, 'w')
    fp.write(json.dumps(docCount) + "\n")
    fp.write(json.dumps(wordSize) + "\n")
    fp.write(json.dumps(wordProb) + "\n")
    fp.write(json.dumps(list(D)) + "\n")
    fp.close()


def normalize(text):
    return text.lower().replace("\n", " ")


def tokenize(text):
    #return filter(lambda i: len(i) > 0, normalize(text).split(' '))
    return normalize(text).split(' ')


def getFeatures(js):
    #text = js['text']
    text = js
    words = tokenize(text)
    return words + [
        "___wordsCount:" + str(len(words)),
        "___linksCount:" + str(text.count("http://")),
        "___mentionCount:" + str(text.count("@")),
        "___hashCount:" + str(text.count("#")),
        #"___source:" + js['source'],
        "___isRt:" + ("1" if 'rt' in words else "0")
    ]

def learn(data_path_spam, data_path_normal):
    """
    This function learns to classify whether a tweet is spam or not based on a Naive Bayes implementation

    :param data_path_spam: Path to the CSV training files containing spam tweets
    :param data_path_normal: Path to the csv containing normal tweets

    :return: None
    """
    spam_tweet_count = 0
    normal_tweet_count = 0

    spam_tweets = []
    normal_tweets = []

    test_data_count = 200

    data_columns = {"id": int, "text": str, "source": str, "user_id": str, "truncated": str, "in_reply_to_status_id": str,
         "in_reply_to_user_id": str, "in_reply_to_screen_name": str, "retweeted_status_id": str, "geo": str,
         "place": str, "contributors": str, "retweet_count": int, "reply_count": str, "favorite_count": str,
         "favorited": str, "retweeted": str, "possibly_sensitive": str, "num_hashtags": int, "num_urls": str,
         "num_mentions": str, "created_at": str, "timestamp": str, "crawled_at": str, "updated": str}

    if not(os.path.exists(data_path_normal) and os.path.exists(data_path_spam)):
        print('A file is not present in the given location')
        return

    spamreader = pd.read_csv(data_path_spam, dtype = data_columns)
    spam_tweets = deepcopy(list(spamreader.get('text')))
    spam_tweet_count = len(spam_tweets)

    print('Reading spam tweets completed, ')
    print('Number of spam tweets: {spam_tweet_count}'.format(spam_tweet_count=spam_tweet_count))

    normal_tweet_reader = pd.read_csv(data_path_normal, dtype=data_columns)
    normal_tweets = deepcopy(list(normal_tweet_reader.get('text')))
    normal_tweet_count = len(normal_tweets)
    print('Reading normal tweets completed')
    print('Number of normal tweets :{normal_tweet_count}'.format(normal_tweet_count=normal_tweet_count))

    training_array = deepcopy(spam_tweets + normal_tweets)
    labels = deepcopy(['B'] * spam_tweet_count + ['G'] * normal_tweet_count)

    # Generate the random numbers for generating a test set
    test_idx = random.sample(range(len(training_array)), test_data_count)
    test_features = []
    test_labels = []
    for idx in test_idx:
        test_features.append(training_array[idx])
        test_labels.append(labels[idx])
        training_array.pop(idx)
        labels.pop(idx)

    D = set()
    wordProb = {'B': defaultdict(int), 'G': defaultdict(int)}
    wordSize = {'B': 0, 'G': 0}
    docCount = {'B': 0, 'G': 0}

    idx = 0
    for line in training_array:
        if type(line) is float:
            continue

        #line = str(line.strip().split("\t"))
        line = remove_url(str(line.strip().split("\t")))
        klass = labels[idx]
        docCount[klass] += 1
        for w in getFeatures(line):
            D.add(w)
            wordProb[klass][w] += 1
            wordSize[klass] += 1

        idx += 1
        if idx % 10000 == 0:
            print(idx)

    writeModel('model.json', docCount, wordSize, wordProb, D)
    print('Training completed')

    model = readModel('model.json')
    correctly_classified = 0
    for idx in range(test_data_count):
        if type(test_features[idx]) is float:
            continue

        prob = classify(model, remove_url(str(test_features[idx])))
        class_ = "G" if prob >= 0 else "B"
        if class_ == test_labels[idx]:
            correctly_classified += 1

        if idx % 10 == 0:
            print('Testing {test_idx}/{length}'.format(test_idx=idx, length=test_data_count))

    print('Accuracy: {correctly_classified}/{test_data_count}'.format(correctly_classified=correctly_classified,
                                                                      test_data_count=test_data_count))

def continue_training(path_to_model, train_path):

    spamreader = pd.read_csv(train_path, dtype=data_columns)
    training_array = deepcopy(list(spamreader.get('text')))
    training_length = len(training_array)
    labels = ['B'] * training_length


    model = readModel(path_to_model)
    (docCount, wordSize, wordProb, D) = model
    D = set(D)
    idx = 0
    for line in training_array:
        if type(line) is float:
            continue
        line = str(line.strip().split("\t"))
        klass = labels[idx]
        docCount[klass] += 1
        for w in getFeatures(line):
            print(w)
            D.add(w)
            wordProb[klass][w] += 1
            wordSize[klass] += 1

        idx += 1
        if idx % 10000 == 0:
            print(idx)

    writeModel('model.json', docCount, wordSize, wordProb, D)

def main():
    spam_tweets_path = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/' \
                          'social_spambots_1.csv/tweets.csv'

    genuine_tweets_path = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/' \
                          'genuine_accounts.csv/tweets.csv'

    train_path = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/tweets.csv'

    spam_test_path = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/tweets.csv'

    train = True

    if train:
        learn(data_path_spam=spam_tweets_path, data_path_normal=genuine_tweets_path)

    #continue_training('model.json', train_path)

    model = readModel('model.json')

    print('Reading file:{file_name}'.format(file_name=spam_test_path))
    test_csv = pd.read_csv(spam_test_path, dtype=data_columns)
    spam_tweets = deepcopy(list(test_csv.get('text')))
    print('Testing spam tweets of length:{length}'.format(length=len(spam_tweets)))
    correctly_classified = 0
    test_idx = 0
    test_length = len(spam_tweets)

    for tweet in spam_tweets:
        test_idx += 1
        val = classify(model, remove_url(tweet))
        if val < 0:
            correctly_classified += 1

        print(val)

        if test_idx % 100 == 0:
            print('Testing {test_idx}/{length}'.format(test_idx=test_idx, length=test_length))
            break

    print('Correctly classified: {correctly_classified}/{total}'.
          format(correctly_classified=correctly_classified, total=test_length))


if __name__ == '__main__':
    main()

