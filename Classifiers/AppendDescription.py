import csv
import re
from tweepy import TweepError

from GetApi import get_api

'''Twitter Account Keys'''
key = ['L5UQsE4pIb9YUJvP7HjHuxSvW',
       'HdRLPYgUqME194Wi2ThbbWfRd9BNHxIr2ms612oX9Yq1QXZdH7',
       '1039011019786342401-iDggGlhErT1KKdVGVXz4Kt7X8v0kIV',
       'MJ17S1uhCaI1zS3NBWksMaWdwjvAjn7cpji5vyhknfcUe']

api = get_api(key[0], key[1], key[2], key[3])


def main():
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/twitter_data/'

    with \
            open(common_path + 'varol2017.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'varol2017-d.csv',
                 'w+',
                 encoding="utf-8") as out:
        reader = csv.DictReader(inp)

        fields = ['id',
                  'description',
                  'bot']

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0

        for row in reader:
            try:
                cnt = cnt + 1
                user = api.get_user(row['id'])
                description = user.description
                if not description:
                    dc_new = ' '
                else:
                    dc = description
                    regex = re.compile('[^a-zA-Z]')
                    # First parameter is the replacement, second parameter is your input string
                    dc_new = regex.sub(' ', dc)
                writer.writerow({'id': row['id'],
                                 'description': dc_new,
                                 'bot': row['bot']})

                print('Row ', cnt, ' written for => ', row['id'])
            except TweepError as e:
                print(e)
                print('skipping')
                continue


if __name__ == '__main__':
    main()
