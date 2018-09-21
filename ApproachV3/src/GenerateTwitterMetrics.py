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


def compute_least_entropy_length(array):
    """
    This function constructs sequences of different lengths and finds the sequence with the least entropy

    :param array: The array containing the sequences

    :return: The length of the sequence and the value of the least entropy
    """

    from nltk import ngrams
    from copy import deepcopy
    import math

    max_length = 0

    # Extract the last value as a predictor
    test_value = array[-1]

    # Move the rest of the array to construct a training array
    train_array = array

    # Count the number of unique sequences
    processed_indices = [False] * len(train_array)

    unique_grams = []
    unique_grams_occurence_count = []
    unique_grams_occurence_total = len(array)
    unique_grams_idx = 0

    # An arbitrarily sufficiently large initial value
    min_entropy = 1000000000000.0

    curr_unique_gram_count = 0
    max_unique_gram_count = 10000000000

    eps = 0.00001
    # Could use a counter and a loop to insert it
    for idx in range(0, len(array)):

        if processed_indices[idx]:
            continue

        print(train_array[idx])

        unique_grams.append(train_array[idx])
        unique_grams_occurence_count.append(1)

        for search_idx in range(idx + 1, len(train_array)):
            #print(train_array[search_idx])
            if processed_indices[search_idx] is False and train_array[idx] == train_array[search_idx]:
                unique_grams_occurence_count[unique_grams_idx] += 1
                processed_indices[search_idx] = 1

        unique_grams_idx += 1


    # Try sequences from length 2 up to length of array - 1
    for curr_length in range(2, len(train_array) - 3):

        grams_condition = [train_array[i:i + curr_length - 1] for i in range(len(train_array) - curr_length)]
        grams = [train_array[i:i + curr_length] for i in range(len(train_array) - curr_length + 1)]

        # Count the number of unique sequences
        processed_indices = [False] * len(grams)

        # Copy the previous unique grams as the unique grams condition of this iteration
        unique_grams_condition = deepcopy(unique_grams)
        unique_grams_condition_occurence_count = deepcopy(unique_grams_occurence_count)
        unique_grams_condition_idx = unique_grams
        unique_grams_condition_occurence_total = unique_grams_occurence_total

        unique_grams = []
        unique_grams_occurence_count = []
        unique_grams_idx = 0
        unique_grams_occurence_total= len(grams)

        curr_entropy = 0.0

        for idx in range(0, len(grams)):

            if processed_indices[idx]:
                continue

            #print(grams[idx])

            unique_grams.append(grams[idx])
            unique_grams_occurence_count.append(1)

            for search_idx in range(idx+1, len(grams)):
                #print(grams[search_idx])
                if processed_indices[search_idx] is False and grams[idx] == grams[search_idx]:
                    unique_grams_occurence_count[unique_grams_idx] += 1
                    processed_indices[search_idx] = 1

            unique_grams_idx += 1

        # Find the conditional probability of each of the sequence
        for prob_idx in range(0, len(unique_grams)):
            unique_gram = unique_grams[prob_idx]
            condition_gram = unique_grams[:-1]
            pos = 0
            for search_idx in range(0, len(unique_grams_condition)):
                if unique_grams_condition[search_idx] == condition_gram:
                    pos = search_idx
                    break
            #print('Unique :' + str(unique_grams_occurence_count[prob_idx]) + 'Condition :' + str(unique_grams_condition_occurence_count[pos]))
            # Find the entropy
            prob_val = (unique_grams_occurence_count[prob_idx] / unique_grams_condition_occurence_total )/ (unique_grams_condition_occurence_count[prob_idx] / unique_grams_condition_occurence_total)
            curr_entropy += -1 * prob_val * math.log(prob_val)

        curr_unique_gram_count = len(unique_grams)
        # Save the sequence that produces the least entropy
        print(curr_entropy)
        print(min_entropy)
        if curr_entropy < min_entropy or (curr_entropy - min_entropy < eps and curr_unique_gram_count > max_unique_gram_count):
            min_entropy = curr_entropy
            max_length = curr_length
            max_unique_gram_count = curr_unique_gram_count

            print(unique_grams)

            # Compute the number of unique sequences too

    return max_length, min_entropy

def main():
     array = [1, 2, 3, 4, 1 , 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
     m, prob = compute_least_entropy_length(array)
     print(m)
     print(prob)


if __name__ ==  '__main__':
    main()