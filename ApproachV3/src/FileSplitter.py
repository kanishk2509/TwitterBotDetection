import csv


def main():
    common_path = '/Users/kanishksinha/PycharmProjects/TwitterBotDetection/ApproachV3/temp_datasets/'

    with \
            open(common_path + 'training_dataset_final.csv',
                 'r+',
                 encoding="utf-8") as inp, \
            open(common_path + 'training_dataset_final_split_1.csv',
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
                  'bot']

        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        cnt = 0

        for row in reader:
            cnt = cnt + 1

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
                             'bot': row['bot']})

            if cnt == 700:
                exit()


if __name__ == '__main__':
    main()
