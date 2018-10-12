import csv
from GetApi import get_api

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

api = get_api(key[0], key[1], key[2], key[3])


def main():
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV4/temp_datasets/'

    with \
            open(common_path + 'training-dataset-final-v4.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'training-dataset-final-v4-d.csv',
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
                  'description',
                  'bot']

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0

        for row in reader:
            cnt = cnt + 1
            user = api.get_user(row['id'])
            description = user.description
            if not description:
                dc = 0
            else:
                dc = description
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
                             'avg_cosine_similarity': row['avg_cosine_similarity'],
                             'avg_tweet_sentiment': row['avg_tweet_sentiment'],
                             'std_deviation_friends': row['std_deviation_friends'],
                             'std_deviation_followers': row['std_deviation_followers'],
                             'unique_urls_ratio': row['unique_urls_ratio'],
                             'tweet_url_similarity': row['tweet_url_similarity'],
                             'user_description_len': row['user_description_len'],
                             'user_description_sentiment': row['user_description_sentiment'],
                             'special_char_in_description': row['special_char_in_description'],
                             'tweet_count': row['tweet_count'],
                             'description': dc,
                             'bot': row['bot']})

            print('Row ', cnt, ' written for => ', row['screen_name'].upper())


if __name__ == '__main__':
    main()
