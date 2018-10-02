import csv
from GetApi import get_api
import tweepy
from ApproachV3.src.metrics import GenerateTwitterMetrics as metrics

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
    user = api_.get_user(id)

    if not user.protected:
        # Iterate through all (1000 max) tweets. items() can take a lower max to limit
        for tweet in tweepy.Cursor(api.user_timeline, id=id, tweet_mode='extended').items(1000):
            tweet_times.append(tweet.created_at)
            tweets_parsed += 1
            txt = tweet._json['full_text']
            tweets.append(txt)

        if tweets_parsed == 0:
            return []

        return tweets, tweet_times

    else:
        print("Protected Account: {}".format(id))
        return []


def main():
    common_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV3/datasets/'

    with \
            open(common_path + 'training_dataset_final_split_1.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'training_dataset_final_cce_split_1.csv',
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
                  'bot']

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0

        for row in reader:
            cnt = cnt + 1

            # Get all the tweets and tweet times of the user
            tweets, tweet_times = get_tweet_and_tweet_times(row['id'], api)

            binned_times, binned_sequence = metrics.generate_binned_array(tweet_times)
            first_order_entropy = metrics.calculate_entropy(binned_array=binned_times)
            max_length, conditional_entropy, perc_unique_strings = \
                metrics.compute_least_entropy_length_non_overlapping(list(binned_sequence))

            cce = conditional_entropy + perc_unique_strings * first_order_entropy
            print('Entropy: ', cce)

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
                             'bot': row['bot']})

            print('Row ', cnt, ' written for => ', row['screen_name'].upper())


if __name__ == '__main__':
    main()
