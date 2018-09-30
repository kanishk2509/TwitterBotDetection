import math
import os
import json
import pandas as pd
import functools
from collections import defaultdict
from copy import deepcopy


def sum(dict):
    return functools.reduce(lambda x, y: x + y, dict.values())


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
    Gp += math.log(float(docCount['G']) / sum(docCount))
    Bp += math.log(float(docCount['B']) / sum(docCount))

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
    total_tweets_per_category = 5000

    import csv
    spam_tweet_count = 0
    normal_tweet_count = 0

    spam_tweets = []
    normal_tweets = []

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
    '''
    #with open(data_path_spam, 'r') as csvfile:
        #spamreader = csv.DictReader(csvfile, delimiter=',')

        for row in spamreader:
    
            spam_tweets.append(row['text'])
            spam_tweet_count += 1
    
            print('Spam Tweet Current Count{spam_tweet_count}'.format(spam_tweet_count=spam_tweet_count))
            if spam_tweet_count >= total_tweets_per_category:
                break
            
    '''
    print('Reading spam tweets completed, ')
    print('Number of spam tweets: {spam_tweet_count}'.format(spam_tweet_count=spam_tweet_count))

    normal_tweet_reader = pd.read_csv(data_path_normal, dtype=data_columns)
    normal_tweets = deepcopy(list(normal_tweet_reader.get('text')))
    normal_tweet_count = len(normal_tweets)
    print('Reading normal tweets completed')
    print('Number of normal tweets :{normal_tweet_count}'.format(normal_tweet_count=normal_tweet_count))

    #with open(data_path_normal, 'r') as csvfile:
        #normal_tweet_reader = csv.DictReader(csvfile, delimiter=',')

        #for row in normal_tweet_reader:

            #normal_tweets.append(row['text'])
            #normal_tweet_count += 1

            #print('Normal Tweet Current Count{normal_tweet_count}'.format(normal_tweet_count=normal_tweet_count))

            #if normal_tweet_count >= total_tweets_per_category:
                #break

    training_array = deepcopy(spam_tweets + normal_tweets)
    classes = deepcopy(['B'] * spam_tweet_count + ['G'] * normal_tweet_count)

    print(training_array[0])

    D = set()
    wordProb = {'B': defaultdict(int), 'G': defaultdict(int)}
    wordSize = {'B': 0, 'G': 0}
    docCount = {'B': 0, 'G': 0}

    idx = 0
    for line in training_array:
        if type(line) is float:
            continue

        parts = str(line).strip().split("\t")
        klass = classes[idx]
        docCount[klass] += 1
        for w in getFeatures(line):
            D.add(w)
            wordProb[klass][w] += 1
            wordSize[klass] += 1

        idx += 1

    writeModel('model.json', docCount, wordSize, wordProb, D)
    print('Training completed')


def main():
    spam_tweets_path = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/' \
                          'social_spambots_1.csv/tweets.csv'

    genuine_tweets_path = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/' \
                          'genuine_accounts.csv/tweets.csv'

    learn(data_path_spam=spam_tweets_path, data_path_normal=genuine_tweets_path)


    model = readModel('model.json')
    print(classify(model, 'win facebook tweets good million'))


if __name__ == '__main__':
    main()

