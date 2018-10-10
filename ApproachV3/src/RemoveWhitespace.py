import csv

common_path = 'https://raw.githubusercontent.com/kanishk2509/TwitterBotDetection/master/kaggle_data/'
file = common_path + 'varol2017.csv'
file_op = common_path + 'varol2017_final.csv'


def main():
    with \
            open(file,
                 'r+',
                 encoding="utf-8") as inp, \
            open(file_op,
                 'w+',
                 encoding="utf-8") as out:
        reader = csv.DictReader(inp)
        my_fields = ['id',
                     'bot']
        writer = csv.DictWriter(out, fieldnames=my_fields)
        writer.writeheader()

        for row in reader:
            writer.writerow({'id': row['id'],
                             'bot': row['bot'].lstrip()})


if __name__ == '__main__':
    main()