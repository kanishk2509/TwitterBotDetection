import csv


def main():
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV3/temp_datasets/'

    with \
            open(common_path + 'training-dataset-final-v4.csv',
                 'r+',
                 encoding="utf-8") as inp:
        reader = csv.DictReader(inp)

        cnt_human = 0
        cnt_bot = 0

        for row in reader:
            if row['bot'] == '1':
                cnt_bot += 1
            else:
                cnt_human += 1

        print("Bots = ", cnt_bot)
        print("Humans = ", cnt_human)


if __name__ == '__main__':
    main()
