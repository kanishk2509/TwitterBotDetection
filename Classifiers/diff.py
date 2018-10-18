import csv


def main():
    # Read in the original and new file
    path_v3 = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV3/temp_datasets/balanced_dataset_v3_des.csv'
    path_merged = '/Users/kanishksinha/Desktop/TwitterBotDetection/Classifiers/output.csv'

    with \
            open(path_v3,
                 'r+',
                 encoding="utf-8") as inp_v3, \
            open(path_merged,
                 'r+',
                 encoding="utf-8") as out_m, \
            open('diff.csv',
                 'w+',
                 encoding="utf-8") as diff:

        readerv3 = csv.DictReader(inp_v3)
        readerm = csv.DictReader(out_m)

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
                  'cce',
                  'spam_ratio',
                  'description',
                  'bot']

        writer = csv.DictWriter(diff, fieldnames=fields)
        writer.writeheader()

        arr = []

        for row in readerm:
            arr.append(row['id'])

        print(len(arr))
        s = set(arr)

        cnt = 0
        for row in readerv3:
            if row['id'] not in s:
                cnt += 1
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
                                 'cce': row['cce'],
                                 'spam_ratio': row['spam_ratio'],
                                 'description': row['description'],
                                 'bot': row['bot']})

        print(cnt)


if __name__ == '__main__':
    main()
