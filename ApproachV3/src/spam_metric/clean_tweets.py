from copy import deepcopy
import os
import pandas as pd
import re
import string

data_columns = {"id": int, "text": str, "source": str, "user_id": str, "truncated": str, "in_reply_to_status_id": str,
                "in_reply_to_user_id": str, "in_reply_to_screen_name": str, "retweeted_status_id": str, "geo": str,
                "place": str, "contributors": str, "retweet_count": int, "reply_count": str, "favorite_count": str,
                "favorited": str, "retweeted": str, "possibly_sensitive": str, "num_hashtags": int, "num_urls": str,
                "num_mentions": str, "created_at": str, "timestamp": str, "crawled_at": str, "updated": str}


def remove_url(tweet):
    """
    Regex based URL removed. Removes all nonwhitespace characters after http until a whitespace is reached
    :param tweet: Tweet to be checked
    :return: Tweet that is substituted with URL in the place of the actual URL
    """
    return re.sub(r"http\S+", "URL", tweet)


def clean_and_write_tweets(path, category):
    """
    Cleans and writes the tweets to a file

    :param path: Path to file
    :param category: Category of the tweet

    :return: None
    """

    table = str.maketrans({key: None for key in string.punctuation})

    test_csv = pd.read_csv(path, dtype=data_columns)
    tweets = deepcopy(list(test_csv.get('text')))

    cleaned_tweets = []
    idx = 0
    for tweet in tweets:

        if type(tweet) is str:

            if len(tweet) == 0:
                continue

            # remove URL
            line = remove_url(str(tweet.strip()))
            # remove non Latin characters
            stripped_text = ''
            for c in line:
                stripped_text += c if len(c.encode(encoding='utf_8')) == 1 else ''

            stripped_text = (stripped_text.translate(table)).strip()
            if len(stripped_text) > 0 and stripped_text.lower() != 'nan':
                cleaned_tweets.append(stripped_text)
                idx += 1

    d = {'tweets': cleaned_tweets, 'category': [category] * len(cleaned_tweets)}
    df = pd.DataFrame(data=d)

    write_path = path[:path.rfind('/')]
    df.to_csv(os.path.join(write_path, 'cleaned_tweets.csv'), index=False)

def main():
    path_spam1 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/tweets.csv'
    path_spam2 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/tweets.csv'
    path_spam3 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/tweets.csv'
    path_spam4 = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/tweets.csv'
    path_list = []

    path_list.append(path_spam1)
    path_list.append(path_spam2)
    path_list.append(path_spam3)
    path_list.append(path_spam4)

    for path in path_list:
        print('Cleaning File:{path}'.format(path=path))
        clean_and_write_tweets(path, 0)

    path_normal = '/home/chris/study/Text Mining Project/corpus/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/tweets.csv'
    print('Cleaning file:{path}'.format(path=path_normal))
    clean_and_write_tweets(path_normal, 1)


if __name__ == "__main__":
    main()