import csv
from ApproachV4.GetApi import get_api
from ApproachV4.GetTweetProperties import get_tweet_semantics

'''
The purpose of this module is to append additional sentimental and semantic properties listed below.
Since we have already mined and generated a training data, we are just appending these properties in order to save time 
instead of starting from the scratch.

    1. Standard deviation of user's friends
    2. Standard deviation of user's followers
    3. Average Frequency of @username mentioned in all the tweets
    4. Average number of "unique" urls in all the tweets, since bots tend to share similar urls over time.
    5. Length of user description
    6. Sentiment of user's description
    7. Special character count in description
    8. Total number of tweets
    
'''

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

api = get_api(key[0], key[1], key[2], key[3])


def main():
    common_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV44/datasets/'

    with \
            open(common_path + 'training_dataset_v200.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'training_dataset_v200f_temp.csv',
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
                  'avg_cosine_sim',
                  'avg_tweet_sentiment',
                  'std_dev_friends',
                  'std_dev_followers',
                  'unique_urls_ratio',
                  'tweet_url_similarity',
                  'user_desc_len',
                  'user_desc_sentiment',
                  'special_char_count',
                  'tweet_count',
                  'bot']

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0

        for row in reader:
            cnt = cnt + 1

            std_dev_friends, \
                std_dev_followers, unique_urls_ratio, \
                tweet_url_similarity, user_desc_len, user_desc_sentiment, \
                special_char_count, tweet_count \
                = get_tweet_semantics(row['id'], api)

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
                             'url_ratio': row['url_ratio'],
                             'avg_cosine_sim': row['avg_cosine_sim'],
                             'avg_tweet_sentiment': row['avg_tweet_sentiment'],
                             'std_dev_friends': std_dev_friends,
                             'std_dev_followers': std_dev_followers,
                             'unique_urls_ratio': unique_urls_ratio,
                             'tweet_url_similarity': tweet_url_similarity,
                             'user_desc_len': user_desc_len,
                             'user_desc_sentiment': user_desc_sentiment,
                             'special_char_count': special_char_count,
                             'tweet_count': tweet_count,
                             'bot': row['bot']})

            print('Row ', cnt, ' written for => ', row['screen_name'].upper())


if __name__ == '__main__':
    main()
