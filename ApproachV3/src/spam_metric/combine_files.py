import os
import pandas as pd

def main():
    path_spam1 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/cleaned_tweets.csv'
    path_spam2 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/cleaned_tweets.csv'
    path_spam3 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/cleaned_tweets.csv'
    path_spam4 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/cleaned_tweets.csv'
    path_normal = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/cleaned_tweets.csv'
    path_list = []

    path_list.append(path_spam1)
    path_list.append(path_spam2)
    path_list.append(path_spam3)
    path_list.append(path_spam4)
    path_list.append(path_normal)

    tweets = []
    categories = []

    dtype = {"tweets": str, "category": int}

    for path in path_list:
        print('Reading file:{path}'.format(path=path))
        data = pd.read_csv(path, dtype=dtype)
        tweets += list(data.get('tweets'))
        categories += list(data.get('category'))

    d = {'tweets': tweets, 'category': categories}
    df = pd.DataFrame(data=d)

    write_path = path[:path.rfind('/')]
    df.to_csv('cleaned_tweets.csv', index=False)


if __name__ == '__main__':
    main()