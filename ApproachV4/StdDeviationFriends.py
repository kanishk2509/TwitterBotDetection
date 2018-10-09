import numpy as np
import tweepy


def compute_std_deviation_friends(user_id, api):
    friends = get_friends_ids(user_id, api)
    return compute(friends)


def get_friends_ids(user_id, api):
    ids = []
    page_count = 0
    for page in tweepy.Cursor(api.friends_ids, id=user_id, count=5000).pages():
        page_count += 1
        print('Getting page {} for friends ids'.format(page_count))
        ids.extend(page)
        if len(ids) > 5000:
            break
    return ids


def compute(friends):
    arr = np.array(friends)
    return np.std(arr)
