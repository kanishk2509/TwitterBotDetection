import csv
import itertools
from GetApi import get_api
from CosineSentenceSimilarity import compute_similarity
import numpy as np
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import tweepy
import re
from textblob import TextBlob

'''
A text mining and pre-processing module extracting features from a user's first 200 tweets such as
1. Average Sentiment
2. Url Ratio
3. Cosine Similarity of tweet pairs
'''

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']


def get_all_tweets(user_id):

    # authorize twitter, initialize tweepy
    api = get_api(key[0], key[1], key[2], key[3])

    # initialize a list to hold all the tweepy Tweets, urls, parsed tweet count
    all_tweets = []
    urls = []
    url_ratio = 0
    tweets_parsed = 0

    try:
        for tweet in tweepy.Cursor(api.user_timeline, id=user_id, tweet_mode='extended').items(200):

            # Fetch tweet's text
            txt = tweet._json['full_text']
            all_tweets.append(txt)

            # Calculate url ratio
            if len(tweet.entities['urls']) > 0:
                for url in tweet.entities['urls']:
                    urls.append(url['expanded_url'])
            tweets_parsed = tweets_parsed + 1
            url_ratio = len(urls) / tweets_parsed

        return all_tweets, url_ratio

    except tweepy.TweepError as e:
        print(e)
        return [], 0.0


def clean_tweet(tweet):
    """
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())


def get_avg_cosine_similarity_and_sentiment(data):
    data_list = list(data)
    data_list_len = len(data_list)
    # Split the data into pairs
    pair_list = list(itertools.combinations(data, 2))
    cosine_sim = 0.0
    total_sentiment = 0.0

    # Compute Average Content Similarity between pairs of tweets over a given range
    # Summation of [similarity(one pair)/(total pairs in list)]
    for pair in pair_list:
        cosine_sim = cosine_sim + compute_similarity(pair[0], pair[1])

    try:
        avg_cosine_sim = cosine_sim / len(pair_list)
    except ZeroDivisionError as e:
        print(e)
        avg_cosine_sim = 1.0

    # Compute Average Tweet Sentiment
    for i in data_list:
        txt = re.sub(r'^http?://.*[\r\n]*', '', i, flags=re.MULTILINE)
        analysis = TextBlob(clean_tweet(txt))
        total_sentiment = total_sentiment + analysis.sentiment.polarity

    avg_sentiment = total_sentiment / data_list_len
    return avg_cosine_sim, avg_sentiment


def main():
    common_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachVx/datasets/'

    with \
            open(common_path + 'training_dataset_final.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'training_dataset_v200a.csv',
                 'w+',
                 encoding="utf-8") as out:

        reader = csv.DictReader(inp)

        fields = ['id', 'id_str', 'screen_name', 'age', 'in_out_ratio',
                  'favorites_ratio', 'status_ratio', 'account_rep', 'avg_tpd',
                  'hashtags_ratio', 'user_mentions_ratio', 'url_ratio',
                  'avg_cosine_sim', 'avg_tweet_sentiment', 'bot']
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        for row in reader:
            sent_array = []
            cnt = cnt + 1
            out_tweets, url_ratio = np.array(get_all_tweets(row['id']))
            if len(out_tweets) > 0:
                for i in out_tweets:
                    # Removing stop words for better analysis
                    word_tokens = word_tokenize(i)
                    filtered_sentence = [w for w in word_tokens if not w in stop_words]
                    filtered_sentence = ''

                    for w in word_tokens:
                        if w not in stop_words:
                            # Stemming the words
                            filtered_sentence = filtered_sentence + ' ' + ps.stem(w)

                    sent_array.append(filtered_sentence)

                cos_sim, avg_sentiment = get_avg_cosine_similarity_and_sentiment(sent_array)

                writer.writerow({'id': row['id'],
                                 'id_str': row['id_str'],
                                 'screen_name': row['screen_name'],
                                 'age': row['age'],
                                 'in_out_ratio': row['in_out_ratio'],
                                 'favorites_ratio': row['favorites_ratio'],
                                 'status_ratio': row['status_ratio'],
                                 'account_rep': row['account_rep'],
                                 'avg_tpd': row['avg_tpd'],
                                 'hashtags_ratio': row['hashtags_ratio'],
                                 'user_mentions_ratio': row['user_mentions_ratio'],
                                 'url_ratio': url_ratio,
                                 'avg_cosine_sim': cos_sim,
                                 'avg_tweet_sentiment': avg_sentiment,
                                 'bot': row['bot']})

                print('Row ', cnt, ' written for => ', row['screen_name'].upper())

            else:
                print("Skipping record due to error...")


if __name__ == '__main__':
    main()
