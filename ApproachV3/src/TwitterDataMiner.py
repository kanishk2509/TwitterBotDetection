import csv
from ApproachV3.src.GetApi import get_api
from ApproachV3.src.GetAccountProperties import get_data

from ApproachV3.src.spam_metric.multinomial_bayes import load

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

vectorizer, classifier = load()


def lookup(user_id):
    api = get_api(key[0], key[1], key[2], key[3])
    X = get_data(user_id, api, vectorizer, classifier)
    return X


def main():
    with \
            open('/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV3/datasets'
                 '/training_dataset_cleaned.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open('/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV3/datasets'
                 '/training_dataset_final_cce_split_3.csv',
                 'w+', 
                 encoding="utf-8") as out:

        reader = csv.DictReader(inp)
        my_fields = ['id',
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
        writer = csv.DictWriter(out, fieldnames=my_fields)
        writer.writeheader()

        for row in reader:
            data = (lookup(row['id'].replace('"', '')))
            if not data:
                print('Skipping record!')
                print(data)
            else:
                print(data)
                writer.writerow({'id': row['id'],
                                 'id_str': str(row['id']),
                                 'screen_name': data[0],
                                 'location': data[13],
                                 'age': data[1],
                                 'in_out_ratio': data[2],
                                 'favorites_ratio': data[3],
                                 'status_ratio': data[4],
                                 'account_rep': data[5],
                                 'avg_tpd': data[6],
                                 'hashtags_ratio': data[7],
                                 'user_mentions_ratio': data[8],
                                 'mal_url_ratio': data[9],
                                 'cce': data[10],
                                 'spam_ratio': data[11],
                                 'bot': row['bot']})


if __name__ == '__main__':
    main()
