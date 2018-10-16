import matplotlib.pyplot as plt
import csv


def main():
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
    sentiment_user = []
    sentiment_bot = []
    tweet_count_user = []
    tweet_count_bot = []
    similarity_user = []
    similarity_bot = []

    print('Main')
    with \
            open('/home/chris/PycharmProjects/TwitterBotDetection/ApproachV4/datasets'
                 '/dataset.csv',
                 'r+',
                 encoding="utf-8") as inp:
            reader = csv.DictReader(inp)

            for row in reader:
                if row['bot'] == '1':
                    hashtags_bot.append(float(row['hashtags_ratio']))
                    usermentions_bot.append(float(row['user_mentions_ratio']))
                    rep_bot.append(float(row['account_rep']))
                    tpd_bot.append(float(row['avg_tpd']))
                    age_bot.append(float(row['age']))
                    sentiment_bot.append(float(row['avg_tweet_sentiment']))
                    similarity_bot.append(float(row['avg_cosine_similarity']))
                    tweet_count_bot.append(float(row['tweet_count']))

                else:
                    hashtags_user.append(float(row['hashtags_ratio']))
                    usermentions_user.append(float(row['user_mentions_ratio']))
                    x = float(row['account_rep'])
                    if x < 0:
                        x = 0
                    rep_user.append(x)
                    tpd_user.append(float(row['avg_tpd']))
                    age_user.append(float(row['age']))
                    sentiment_user.append(float(row['avg_tweet_sentiment']))
                    similarity_user.append(float(row['avg_cosine_similarity']))
                    tweet_count_user.append(float(row['tweet_count']))

    x = range(100)
    y = range(100, 200)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('similarity')
    plt.ylabel('tweet count')

    ax1.scatter(similarity_bot, tweet_count_bot, s=10, c='b', marker="s", label='bot')
    ax1.scatter(similarity_user, tweet_count_user, s=10, c='r', marker="o", label='normal')

    #ax1.scatter(cce_array_bot, spam_ratio_array_bot, s=10, c='b', marker="s", label='bot')
    #ax1.scatter(cce_array_user, spam_ratio_array_user, s=10, c='r', marker="o", label='normal')

    plt.legend(loc='upper left');
    plt.show()

if __name__ == '__main__':
    main()