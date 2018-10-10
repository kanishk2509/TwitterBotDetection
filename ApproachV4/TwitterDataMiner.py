import csv
from GetApi import get_api
import numpy as np
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import tweepy
from datetime import datetime
from ApproachV4.CosineSentenceSimilarity import get_avg_cosine_similarity
from ApproachV4.AverageSentenceSentiment import get_avg_sentiment
from ApproachV4.GetTweetProperties import get_tweet_semantics

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

api = get_api(key[0], key[1], key[2], key[3])


def get_all_tweets(user_id):
    # initialize a list to hold all the tweepy Tweets, urls, parsed tweet count
    all_tweets = []
    urls = []
    url_ratio = 0
    tweets_parsed = 0

    try:
        for tweet in tweepy.Cursor(api.user_timeline, id=user_id, tweet_mode='extended').items(1000):

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


def main():
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV4/datasets/'

    with \
            open(common_path + 'dataset-2-temp.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'dataset-2-1-temp.csv',
                 'w+',
                 encoding="utf-8") as out:

        reader = csv.DictReader(inp)

        fields = ['id',
                  'id_str',
                  'screen_name',
                  'age',
                  'in_out_ratio',
                  'favorites_ratio',
                  'status_ratio',
                  'account_rep',
                  'avg_tpd',
                  'hashtags_ratio',
                  'user_mentions_ratio',
                  'url_ratio',
                  'avg_cosine_similarity',
                  'avg_tweet_sentiment',
                  'std_deviation_friends',
                  'std_deviation_followers',
                  'unique_urls_ratio',
                  'tweet_url_similarity',
                  'user_description_len',
                  'user_description_sentiment',
                  'special_char_in_description',
                  'tweet_count',
                  'bot']

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        for row in reader:
            cnt = cnt + 1
            t1 = datetime.now()
            print(cnt, ')')
            print('Getting data for user -> ', row['screen_name'].upper())

            user_id = row['id']
            sent_array = []
            out_tweets, url_ratio = np.array(get_all_tweets(user_id))

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

                cos_sim = get_avg_cosine_similarity(sent_array, 'tweets')
                avg_sentiment = get_avg_sentiment(sent_array)

                tbl = get_tweet_semantics(user_id, api)

                if len(tbl) > 0:
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
                                     'avg_cosine_similarity': cos_sim,
                                     'avg_tweet_sentiment': avg_sentiment,
                                     'std_deviation_friends': tbl[1],
                                     'std_deviation_followers': tbl[2],
                                     'unique_urls_ratio': tbl[3],
                                     'tweet_url_similarity': tbl[4],
                                     'user_description_len': tbl[5],
                                     'user_description_sentiment': tbl[6],
                                     'special_char_in_description': tbl[7],
                                     'tweet_count': tbl[8],
                                     'bot': row['bot']})

                    print('Row ', cnt, ' written for => ', row['screen_name'].upper())
                    t2 = datetime.now() - t1
                    print('Time elapsed:', t2)

                else:
                    cnt = cnt - 1
                    print("Writing null values due to error...")
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
                                     'avg_cosine_similarity': cos_sim,
                                     'avg_tweet_sentiment': avg_sentiment,
                                     'std_deviation_friends': 'nan',
                                     'std_deviation_followers': 'nan',
                                     'unique_urls_ratio': 'nan',
                                     'tweet_url_similarity': 'nan',
                                     'user_description_len': 'nan',
                                     'user_description_sentiment': 'nan',
                                     'special_char_in_description': 'nan',
                                     'tweet_count': 'nan',
                                     'bot': row['bot']})

            else:
                cnt = cnt - 1
                print("No tweets for user...")


if __name__ == '__main__':
    main()
