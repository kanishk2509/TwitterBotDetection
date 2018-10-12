import itertools
import math
import re
from collections import Counter

WORD = re.compile(r'\w+')


def compute_similarity(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    return get_cosine(vector1, vector2)


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def get_avg_cosine_similarity(data, type):
    # Split the data into pairs
    pair_list = list(itertools.combinations(data, 2))
    cosine_sim = 0.0

    # Compute Average Content Similarity between pairs of tweets over a given range
    # Summation of [similarity(one pair)/(total pairs in list)]
    for pair in pair_list:
        cosine_sim = cosine_sim + compute_similarity(pair[0], pair[1])

    try:
        avg_cosine_sim = cosine_sim / len(pair_list)
    except ZeroDivisionError as e:
        print(e)
        avg_cosine_sim = 1.0

    print('Average Similarity in ', type, " : ", avg_cosine_sim)

    return avg_cosine_sim
