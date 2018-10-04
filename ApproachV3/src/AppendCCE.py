import csv
import time
import numpy as np
from GetApi import get_api
import tweepy
from ApproachV3.src.metrics import GenerateTwitterMetrics as metrics
from ApproachV3.src.spam_metric.multinomial_bayes import load, preprocess

'''
The purpose of this module is to append CCE to the existing training data.
Since we have already mined and generated a training data, we are just appending these properties in order to save time 
instead of starting from the scratch.

'''

'''Twitter Account Keys'''
key = ['76vFgL9yGv0sW6gdZs1IsXA6q',
       'dTa5CdXZWoWg02HSg6tZfQEloYaGDxI4xi0Wxiyk2wiByjXqLC',
       '127268392-PQ5EpiFdYjKodmAIvHh98hdfOrcvHLyVJm9E3tLe',
       '4uAWJCqAqRR4JPWeZH9lN70sIuauXs5eTHKe50w2N7A16']

api = get_api(key[0], key[1], key[2], key[3])


def get_tweet_and_tweet_times(id, api_):
    tweet_times = []
    tweets = []
    tweets_parsed = 0
    user = api_.get_user(id)

    if not user.protected:
        # Iterate through all (1000 max) tweets. items() can take a lower max to limit
        for tweet in tweepy.Cursor(api.user_timeline, id=id, tweet_mode='extended').items(1000):
            tweet_times.append(tweet.created_at)
            tweets_parsed += 1
            txt = tweet._json['full_text']
            tweets.append(txt)

        #if tweets_parsed == 0:
            #return tweets, tweet_times

        return tweets, tweet_times

    else:
        print("Protected Account: {}".format(id))
        return tweets, tweet_times


def main():
    common_path = '/home/chris/PycharmProjects/TwitterBotDetection/ApproachV3/datasets/'

    with \
            open(common_path + 'training_dataset_final_split_1.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'training_dataset_final_cce_split_7.csv',
                 'w+',
                 encoding="utf-8") as out:
        reader = csv.DictReader(inp)

        fields = ['id',
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

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0

        vectorizer, classifier = load()

        for row in reader:
            start = time.time()
            cnt = cnt + 1
            print(cnt, ')  Getting tweets for user -> ', row['screen_name'])
            preprocessed = []

            # Get all the tweets and tweet times of the user
            id = row['id']
            # Sometimes ID will be written in exponential form
            if id != row['id_str'][1:-1]:
                id = row['id_str'][1:-1]

            try:
                tweets, tweet_times = get_tweet_and_tweet_times(id, api)
            except tweepy.TweepError as e:
                print(e.reason) # prints 34
                print('Error for user {name}:{reason}'.format(name=row['screen_name'], reason=e.reason))
                continue

            # If there are no tweets move to next
            if len(tweets) <= 50 or len(tweet_times) == 0:
                print('Not enough tweet data. skipping record id {id}'.format(id=row['screen_name']))
                continue

            binned_times, binned_sequence = metrics.generate_binned_array(tweet_times)
            first_order_entropy = metrics.calculate_entropy(binned_array=binned_times)
            max_length, conditional_entropy, perc_unique_strings = \
                metrics.compute_least_entropy_length_non_overlapping(list(binned_sequence))

            cce = conditional_entropy + perc_unique_strings * first_order_entropy

            for tweet in tweets:
                preprocessed_tweet = (preprocess(tweet)).strip()
                if len(preprocessed_tweet) > 0:
                    preprocessed.append(preprocess(tweet))

            # if the user has no Latin alphabet tweets, assign a probability of 0.5 to spam ratio
            if len(preprocessed) > 1:
                vectorized_tweet = vectorizer.transform(preprocessed)
                prediction = classifier.predict(vectorized_tweet)

                length = len(prediction)
                spam_ratio = (length - sum(prediction))/length

            else:
                prediction = []
                spam_ratio = 0.5

            print('Entropy: ', cce)
            #print(prediction)
            print('Spam Ratio: ', spam_ratio)

            writer.writerow({'id': row['id'],
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
                             'bot': row['bot']})

            print('Row ', cnt, ' written for => ', row['screen_name'].upper())
            end = time.time()
            print('Elapsed time:{elapsed}'.format(elapsed = (end-start)))


if __name__ == '__main__':
    main()
