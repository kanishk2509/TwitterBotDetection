import pandas as pd

import csv
from tweepy import TweepError
from numpy import genfromtxt


def main():
    path_v3 = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV3/temp_datasets/balanced_dataset_v3_des.csv'
    path_v4 = '/Users/kanishksinha/Desktop/TwitterBotDetection/ApproachV4/temp_datasets/balanced_dataset_v4.csv'
    merged_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/Classifiers/merged_dataset.csv'

    a = pd.read_csv(path_v3)
    b = pd.read_csv(path_v4)
    merged = a.merge(b, on='id')
    merged.to_csv("output.csv", index=False)

    merged_cols = ['id',
                   'screen_name',
                   'age',
                   'in_out_ratio',
                   'favorites_ratio',
                   'status_ratio',
                   'account_rep',
                   'avg_tpd',
                   'hashtags_ratio',
                   'user_mentions_ratio',
                   'url_ratio',
                   'cce',
                   'spam_ratio',
                   'avg_cosine_similarity',
                   'avg_tweet_sentiment',
                   'std_deviation_friends',
                   'std_deviation_followers',
                   'unique_urls_ratio',
                   'tweet_url_similarity',
                   'user_description_len',
                   'user_description_sentiment',
                   'special_char_in_description',
                   'tweet_count',
                   'description',
                   'bot']


if __name__ == '__main__':
    main()
