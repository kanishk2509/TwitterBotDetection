import re
from textblob import TextBlob


def get_avg_sentiment(data):
    data_list = list(data)
    data_list_len = len(data_list)
    total_sentiment = 0.0
    print("Calculating sentiment for ", data_list_len, " tweets")

    if data_list_len > 0:
        # Compute Average Tweet Sentiment
        for i in data_list:
            txt = re.sub(r'^http?://.*[\r\n]*', '', i, flags=re.MULTILINE)
            analysis = TextBlob(clean_tweet(txt))
            total_sentiment = total_sentiment + analysis.sentiment.polarity

        avg_sentiment = total_sentiment / data_list_len
    else:
        avg_sentiment = 0

    print("Avg Sentiment for tweets: ", avg_sentiment)

    return avg_sentiment


def get_avg_sentiment_single(data, type):
    total_sentiment = 0.0

    txt = re.sub(r'^http?://.*[\r\n]*', '', data, flags=re.MULTILINE)
    analysis = TextBlob(clean_tweet(txt))
    total_sentiment = total_sentiment + analysis.sentiment.polarity
    print("Average sentiment for ", type, ": ", total_sentiment)

    return total_sentiment


def clean_tweet(tweet):
    """
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())
