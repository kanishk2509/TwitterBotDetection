"""
This file gets the times of all tweets of an user and then generates a binned array based on the time of tweets
"""

from collections import Counter
import numpy as np


def generate_binned_array(tweet_times):
    """
    The tweet times array gives the times of each of the users tweets

    :param tweet_times: The times of the tweets of the user

    :return: An array containing a vector of binned tweet times
    """

    # The number of bins to be created for the whole array
    num_bins = 100

    tweet_time_diff = compute_time_difference_between_tweets(tweet_times)

    # Convert the tweets to meaningful numerical representations

    bins = np.linspace(min(tweet_time_diff), max(tweet_time_diff), num_bins)

    digitized = np.digitize(tweet_time_diff, bins)

    binned_elements_count = dict(Counter(digitized))

    binned_array = list()

    for idx in range(1, num_bins):
        binned_array.append(binned_elements_count.get(idx, 0))

    return binned_array


def compute_time_difference_between_tweets(tweet_times):
    """
    This function computes the time intervals between two successive tweet times
    The times should be absolute times in milli seconds

    :param tweet_times: Time of successive tweets in milli seconds

    :return: Time interval between two tweets
    """
    import datetime

    intervals = list()
    for idx in range(0, len(tweet_times) - 1):
        # Convert to epoch time and find the difference
        intervals[idx] = datetime.datetime(tweet_times[idx]).timestamp() - \
                         datetime.datetime(tweet_times[idx+1]).timestamp()

    return intervals
