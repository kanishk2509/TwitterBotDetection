from collections import Counter
import math


def calculate_entropy(binned_array):
    """
    This function calculates the total entropy of the array

    :param binned_array: A binned array containing Q bins

    :return: The total entropy of the array
    """

    if len(binned_array <= 1):
        return 0

    total_count = len(binned_array)

    counted_elements = dict(Counter(binned_array))

    total_entropy = 0.0

    for element in counted_elements:

        count = counted_elements.get(element)

        probability = count / total_count

        total_entropy += -1 * probability * math.log(probability)

    return total_entropy


