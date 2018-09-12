import csv
import re


def main():
    with \
            open('training_dataset.csv', 'r+', encoding="utf-8") as inp, \
            open('training_dataset_cleaned.csv', 'w+', encoding="utf-8") as out:

        reader = csv.DictReader(inp)
        my_fields = ['id', 'id_str', 'screen_name', 'location', 'Age', 'In-out-ratio', 'favorites-ratio',
                     'Status-ratio',
                     'Account-rep', 'Avg-tpd', 'Hashtags-ratio', 'User-mentions-ratio', 'Mal-url-ratio', 'Url-ratio',
                     'bot']
        writer = csv.DictWriter(out, fieldnames=my_fields)
        writer.writeheader()

        for row in reader:
            print(row)
            regex = re.compile('[+@_!#$%^&*()<>?/\|}{~:]')
            id_str = row['id_str'].replace('"', '')
            if regex.search(id_str) is None:
                writer.writerow(row)
            else:
                print("skipping this ", id_str)


if __name__ == '__main__':
    main()
