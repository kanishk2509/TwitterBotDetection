import csv
from datetime import datetime

import numpy as np

from AverageSentenceSentiment import get_avg_sentiment
from CosineSentenceSimilarity import get_avg_cosine_similarity
from GetApi import get_api
import tweepy
from nltk import word_tokenize
from sklearn.feature_extraction import stop_words

from ApproachV3.src.metrics import GenerateTwitterMetrics as metrics
from ApproachV3.src.spam_metric.multinomial_bayes import load, preprocess
from GetTweetProperties import get_tweet_semantics
from TwitterDataMiner import get_all_tweets

'''
The purpose of this module is to append CCE to the existing training data.
Since we have already mined and generated a training data, we are just appending these properties in order to save time 
instead of starting from the scratch.

'''

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

api = get_api(key[0], key[1], key[2], key[3])


def get_tweet_and_tweet_times(id, api_):
    tweet_times = []
    tweets = []
    tweets_parsed = 0
    try:
        user = api_.get_user(id)
    except tweepy.TweepError as e:
        print(e)
        return [], []

    if not user.protected:
        # Iterate through all (1000 max) tweets. items() can take a lower max to limit
        for tweet in tweepy.Cursor(api.user_timeline, id=id, tweet_mode='extended').items(1000):
            tweet_times.append(tweet.created_at)
            tweets_parsed += 1
            txt = tweet._json['full_text']
            tweets.append(txt)

        if tweets_parsed == 0:
            return [], []

        return tweets, tweet_times

    else:
        print("Protected Account: {}".format(id))
        return [], []


def main():
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/kaggle_data/test_datasets/'

    with \
            open(common_path + 'test_data_temp_copy.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'test-dataset-final-v3.csv',
                 'w+',
                 encoding="utf-8") as out_v3, \
            open(common_path + 'test-dataset-final-v4.csv',
                 'w+',
                 encoding="utf-8") as out_v4:
        reader = csv.DictReader(inp)

        fields_v3 = ['id',
                     'id_str',
                     'screen_name',
                     'location',
                     'age',
                     'in_out_ratio',
                     'favorites_ratio',
                     'status_ratio',
                     'account_rep',
                     'avg_tpd',
                     'hashtags_ratio',
                     'user_mentions_ratio',
                     'mal_url_ratio',
                     'cce',
                     'spam_ratio',
                     'bot']

        fields_v4 = ['id',
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

        writer_v3 = csv.DictWriter(out_v3, fieldnames=fields_v3)
        writer_v4 = csv.DictWriter(out_v4, fieldnames=fields_v4)

        writer_v3.writeheader()
        writer_v4.writeheader()

        cnt = 0

        vectorizer, classifier = load()

        for row in reader:
            t1 = datetime.now()

            # Generate v3 dataset
            preprocessed = []
            cnt = cnt + 1
            print(cnt, ')')
            print('Getting tweets for user -> ', row['screen_name'])

            # Get all the tweets and tweet times of the user
            tweets, tweet_times = get_tweet_and_tweet_times(row['id'], api)

            if len(tweets) == 0 or len(tweet_times) == 0:
                print('Above error, skipping record...')
                cnt = cnt - 1
                continue

            if len(tweets) <= 50:
                print('Not enough tweet data, skipping record...')
                cnt = cnt - 1
                continue

            print('Computing Entropy...')
            binned_times, binned_sequence = metrics.generate_binned_array(tweet_times)
            first_order_entropy = metrics.calculate_entropy(binned_array=binned_times)
            max_length, conditional_entropy, perc_unique_strings = \
                metrics.compute_least_entropy_length_non_overlapping(list(binned_sequence))

            cce = conditional_entropy + perc_unique_strings * first_order_entropy

            print('Entropy: ', cce)

            for tweet in tweets:
                if not tweet:
                    continue
                preprocessed.append(preprocess(tweet).strip())

            if len(preprocessed) > 1:
                vectorized_tweet = vectorizer.transform(preprocessed)
                prediction = classifier.predict(vectorized_tweet)
                spam_ratio = (len(prediction) - sum(prediction)) / len(prediction)
            else:
                prediction = []
                spam_ratio = 0.5

            print('Spam Ratio: ', spam_ratio)

            writer_v3.writerow({'id': row['id'],
                                'id_str': row['id_str'],
                                'screen_name': row['screen_name'],
                                'location': row['location'],
                                'age': row['age'],
                                'in_out_ratio': row['in_out_ratio'],
                                'favorites_ratio': row['favorites_ratio'],
                                'status_ratio': row['status_ratio'],
                                'account_rep': row['account_rep'],
                                'avg_tpd': row['avg_tpd'],
                                'hashtags_ratio': row['hashtags_ratio'],
                                'user_mentions_ratio': row['user_mentions_ratio'],
                                'mal_url_ratio': row['mal_url_ratio'],
                                'cce': cce,
                                'spam_ratio': spam_ratio,
                                'bot': 0})

            # Generate v4 dataset
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
                    writer_v4.writerow({'id': row['id'],
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
                                        'bot': 0})

                else:
                    cnt = cnt - 1
                    print("Writing null values due to error...")
                    writer_v4.writerow({'id': row['id'],
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
                                        'bot': 0})

            else:
                cnt = cnt - 1
                print("No tweets for user...")

            print('Row ', cnt, ' written for => ', row['screen_name'].upper())
            t2 = datetime.now() - t1
            print('Time elapsed:', t2)


if __name__ == '__main__':
    main()
