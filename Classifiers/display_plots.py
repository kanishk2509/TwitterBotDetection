import matplotlib.pyplot as plt
import csv


def main():
    cce_array_user = []
    cce_array_bot = []
    spam_ratio_array_user = []
    spam_ratio_array_bot = []
    hashtags_user = []
    hashtags_bot = []
    usermentions_user = []
    usermentions_bot = []
    rep_user = []
    rep_bot = []
    tpd_user = []
    tpd_bot = []
    age_user = []
    age_bot = []

    print('Main')
    with \
            open('/home/chris/PycharmProjects/TwitterBotDetection/ApproachV3/datasets'
                 '/completed_dataset.csv',
                 'r+',
                 encoding="utf-8") as inp:
            reader = csv.DictReader(inp)

            for row in reader:
                if row['bot'] == '1':
                    cce_array_bot.append(float(row['cce']))
                    spam_ratio_array_bot.append(float(row['spam_ratio']))
                    hashtags_bot.append(float(row['hashtags_ratio']))
                    usermentions_bot.append(float(row['user_mentions_ratio']))
                    rep_bot.append(float(row['account_rep']))
                    tpd_bot.append(float(row['avg_tpd']))
                    age_bot.append(float(row['age']))


                else:
                    cce_array_user.append(float(row['cce']))
                    spam_ratio_array_user.append(float(row['spam_ratio']))
                    hashtags_user.append(float(row['hashtags_ratio']))
                    usermentions_user.append(float(row['user_mentions_ratio']))
                    x = float(row['account_rep'])
                    if x < 0:
                        x = 0
                    rep_user.append(x)
                    tpd_user.append(float(row['avg_tpd']))
                    age_user.append(float(row['age']))

    x = range(100)
    y = range(100, 200)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('rep')
    plt.ylabel('age')

    ax1.scatter(rep_bot, age_bot, s=10, c='b', marker="s", label='bot')
    ax1.scatter(rep_user, age_user, s=10, c='r', marker="o", label='normal')

    #ax1.scatter(cce_array_bot, spam_ratio_array_bot, s=10, c='b', marker="s", label='bot')
    #ax1.scatter(cce_array_user, spam_ratio_array_user, s=10, c='r', marker="o", label='normal')

    plt.legend(loc='upper left');
    plt.show()

if __name__ == '__main__':
    main()