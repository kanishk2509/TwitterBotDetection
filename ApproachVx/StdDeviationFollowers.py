import numpy as np
import tweepy


def compute_std_deviation_followers(user_id, api):
    followers = get_followers_ids(user_id, api)
    return compute(followers)


def get_followers_ids(user_id, api):
    ids = []
    page_count = 0
    for page in tweepy.Cursor(api.followers_ids, id=user_id, count=5000).pages():
        page_count += 1
        print('Getting page {} for followers ids'.format(page_count))
        ids.extend(page)
        if len(ids) > 5000:
            break
    return ids


def compute(followers):
    arr = np.array(followers)
    return np.std(arr)
