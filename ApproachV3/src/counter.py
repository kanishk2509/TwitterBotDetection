import csv


def main():
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/Classifiers/final_merged.csv'

    with \
            open(common_path,
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
