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

    max_prob = 0
    max_length = 0
    curr_length = 0
    curr_prob = 0

    # Extract the last value as a predictor
    test_value = array[-1]

    # Move the rest of the array to construct a training array
    train_array = array

    # Try sequences from length 2 up to length of array - 1
    for curr_length in range(2, len(train_array)-1):
        '''
        # Transform the array into n grams
        #ngrams_prev = ngrams(train_array, curr_length - 1)
        #ngrams_curr = ngrams(train_array, curr_length)

        grams_condition = [train_array[i:i + curr_length - 1] for i in range(len(train_array) - curr_length)]
        grams = [train_array[i:i + curr_length] for i in range(len(train_array) - curr_length + 1)]

        # find the most common occurence in the array for computing the probability
        conditional_variables = train_array[curr_length - 1:]

        occurence_count = dict(Counter(conditional_variables))

        max_items_key = []
        max_occurence = 0
        for item in occurence_count:
            count = occurence_count.get(item)
            if count > max_occurence:
                max_occurence = count
                max_items_key.clear()
                max_items_key.append(item)

            elif count == max_occurence:
                max_items_key.append(item)

        # Compute the probability for elements in the max occurence list
        # Probability is computed by counting the number of conditional sequences / t

        #Find the occurences of all sequences [train_array, test_value] to the probability of occurences of test value

        # Predict the probability value using the test_value
        '''
        grams_condition = [train_array[i:i + curr_length - 1] for i in range(len(train_array) - curr_length)]
        grams = [train_array[i:i + curr_length] for i in range(len(train_array) - curr_length + 1)]

        # Count the number of unique sequences
        processed_indices = [False] * len(grams)
        unique_grams = []
        unique_grams_occurence_count = []
        unique_grams_idx = 0

        for idx in range(0, len(grams)):

            if processed_indices[idx]:
                continue

            print(grams[idx])

            unique_grams.append(grams[idx])
            unique_grams_occurence_count.append(1)

            for search_idx in range(idx+1, len(grams)):
                print(grams[search_idx])
                if processed_indices[search_idx] is False and grams[idx] == grams[search_idx]:
                    unique_grams_occurence_count[unique_grams_idx] += 1
                    processed_indices[search_idx] = 1

            unique_grams_idx += 1

        # Find the conditions of each of the unique sequences and count them.

        # Find the conditional probability of each of the sequence

        # Find the entropy


        if curr_prob > max_prob:
            max_prob = curr_prob
            max_length = curr_length

            # Compute the number of unique sequences too

    return max_length, max_prob

def main():
     array = [1, 2, 3, 4, 1 , 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
     m, prob = compute_least_entropy_length(array)
     print(m)
     print(prob)


if __name__ ==  '__main__':
    main()