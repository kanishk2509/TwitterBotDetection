import csv
import collections
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


def main():
    str_des = ''
    common_path = '/Users/kanishksinha/Desktop/TwitterBotDetection/Classifiers/'
    with \
            open(common_path + 'varol2017-d.csv',
                 'r+',
                 encoding="utf-8") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            str_des = str_des + row['description']
            print(str_des)


    # Instantiate a dictionary, and for every word in the file,
    # Add to the dictionary if it doesn't exist. If it does, increase the count.
    wordcount = {}
    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.
    stop_words = set(stopwords.words('english'))

    for word in str_des.lower().split():
        word = word.replace(".", "")
        word = word.replace(",", "")
        word = word.replace(":", "")
        word = word.replace("\"", "")
        word = word.replace("!", "")
        word = word.replace("â€œ", "")
        word = word.replace("â€˜", "")
        word = word.replace("*", "")
        if word not in stop_words:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    # Print most common word
    word_counter = collections.Counter(wordcount)
    for word, count in word_counter.most_common(100):
        print(word, ": ", count)
    # Create a data frame of the most common words
    # Draw a bar chart
    lst = word_counter.most_common(100)
    df = pd.DataFrame(lst, columns=['Word', 'Count'])
    df.plot.bar(x='Word', y='Count')


if __name__ == '__main__':
    main()
