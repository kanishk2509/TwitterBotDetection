import itertools

import tweepy
import datetime as dt
import numpy as np
import requests
from CosineSentenceSimilarity import compute_similarity
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import re
from textblob import TextBlob

dow_ratios = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

'''
Step 3
Calculating Account Properties Component
'''


def get_data(user_id, api):
    tbl = []
    try:
        tbl = mine_data(user_id, api)
        return tbl
    except tweepy.TweepError as e:
        print(e)
        return tbl


def get_avg_cosine_similarity_and_sentiment(data):
    data_list = list(data)
    data_list_len = len(data_list)
    # Split the data into pairs
    pair_list = list(itertools.combinations(data, 2))
    cosine_sim = 0.0
    total_sentiment = 0.0

    # Compute Average Content Similarity between pairs of tweets over a given range
    # Summation of [similarity(one pair)/(total pairs in list)]
    for pair in pair_list:
        cosine_sim = cosine_sim + compute_similarity(pair[0], pair[1])

    try:
        avg_cosine_sim = cosine_sim / len(pair_list)
    except ZeroDivisionError as e:
        print(e)
        avg_cosine_sim = 1.0

    print('Average Similarity in tweets :: ', avg_cosine_sim)

    # Compute Average Tweet Sentiment
    for i in data_list:
        txt = re.sub(r'^http?://.*[\r\n]*', '', i, flags=re.MULTILINE)
        analysis = TextBlob(clean_tweet(txt))
        total_sentiment = total_sentiment + analysis.sentiment.polarity

    avg_sentiment = total_sentiment / data_list_len
    print('Average Sentiment for this user :: ', avg_sentiment)
    return avg_cosine_sim, avg_sentiment


def get_all_tweets_related_properties(user_id, api, user, bot_flag):

    # initialize a list to hold all the tweepy Tweets, urls, parsed tweet count
    global cur_date
    tbl = []
    sent_array = []
    all_tweets = []
    urls = []
    tweets_parsed = 0
    tweets_per_day = [-1]
    hashtags_recorded = 0
    user_mentions_recorded = 0
    date_count = 0

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    if not user.protected:

        try:
            for tweet in tweepy.Cursor(api.user_timeline, id=user_id, tweet_mode='extended').items(1000):

                # Fetch tweet's text
                txt = tweet._json['full_text']
                all_tweets.append(txt)

                # Calculate url ratio
                if len(tweet.entities['urls']) > 0:
                    for url in tweet.entities['urls']:
                        urls.append(url['expanded_url'])

                # If this tweet contains hashtags, count them
                if len(tweet.entities['hashtags']) > 0:
                    hashtags_recorded += len(tweet.entities['hashtags'])

                # If this tweet contained user mentions, count them
                if len(tweet.entities['user_mentions']) > 0:
                    user_mentions_recorded += len(tweet.entities['user_mentions'])

                # Count up the tweets for each day
                # First if block captures date of most recent tweet
                if date_count == 0:
                    cur_date = tweet.created_at
                    tweets_per_day.append(0)
                    date_count += 1
                    tweets_per_day[date_count] += 1
                # elif block handles first tweet of next day
                elif tweet.created_at.day != cur_date.day:
                    cur_date = tweet.created_at
                    date_count += 1
                    tweets_per_day.append(0)
                    tweets_per_day[date_count] += 1
                # Else block handles more tweets on the same day
                else:
                    tweets_per_day[date_count] += 1

                tweets_parsed = tweets_parsed + 1

            url_ratio = len(urls) / tweets_parsed
            print("Url Ratio :: ", url_ratio)

            # Calculate ratio of total hashtags over tweets parsed
            hashtags_ratio = hashtags_recorded / tweets_parsed
            print("Hashtags Ratio :: ", hashtags_ratio)

            # Calculate ratio of total user mentions over tweets parsed
            user_mentions_ratio = user_mentions_recorded / tweets_parsed
            print("User Mentions Ratio :: ", user_mentions_ratio)

            # Slice the tweets_per_day list to remove the first -1 value
            tweets_per_day = tweets_per_day[1:]
            avg_tpd = np.average(tweets_per_day)
            print("Average Tweet/Day: ", avg_tpd)

            tbl.append(avg_tpd)
            tbl.append(hashtags_ratio)
            tbl.append(user_mentions_ratio)
            tbl.append(url_ratio)

            if len(all_tweets) > 0:
                for i in all_tweets:
                    # Removing stop words for better analysis
                    word_tokens = word_tokenize(i)
                    filtered_sentence = ''
                    for w in word_tokens:
                        if w not in stop_words:
                            # Stemming the words
                            filtered_sentence = filtered_sentence + ' ' + ps.stem(w)

                    sent_array.append(filtered_sentence)

                cos_sim, avg_sentiment = get_avg_cosine_similarity_and_sentiment(sent_array)

            else:
                cos_sim, avg_sentiment = 0.0, 0.0

            '''if len(urls) > 0:
                        mal_urls_ratio = num_malicious_urls(urls) / len(urls)
                        print("mal_urls_ratio: ", mal_urls_ratio)'''

            tbl.append(cos_sim)
            tbl.append(avg_sentiment)
            tbl.append(bot_flag)

            return tbl

        except tweepy.TweepError as e:
            print(e)
            return []

    else:
        print("Protected Account: {}".format(user_id))
        return []


def clean_tweet(tweet):
    """
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())


def mine_data(user_id, api):

    tbl = []

    user = api.get_user(user_id)

    print('User Screen Name :: ', user.screen_name)
    regexp = re.compile(r'Bot|bot|b0t|B0T|B0t')
    if regexp.search(user.screen_name.lower()):
        bot_flag = 1
    else:
        bot_flag = 0
    age = dt.datetime.today().timestamp() - user.created_at.timestamp()
    print("User Age :: ", age, " seconds")

    in_out_ratio = 1
    if user.friends_count != 0:
        in_out_ratio = user.followers_count / user.friends_count

    favourites_ratio = 86400 * user.favourites_count / age
    print("Favourites Ratio :: ", favourites_ratio)

    status_ratio = 86400 * user.statuses_count / age
    print("Status Ratio :: ", status_ratio)

    acct_rep = 0
    if user.followers_count + user.friends_count != 0:
        acct_rep = user.followers_count / (user.followers_count + user.friends_count)
        print("Account Reputation :: ", acct_rep)

    symbols = r'_|%|"| '
    # screen_name_binary = user.screen_name.contains(symbols, case=False, na=False)
    tbl.append(user_id)
    # tbl.append(screen_name_binary)
    tbl.append(age)
    tbl.append(in_out_ratio)
    tbl.append(favourites_ratio)
    tbl.append(status_ratio)
    tbl.append(acct_rep)

    tbl2 = get_all_tweets_related_properties(user_id, api, user, bot_flag)

    for i in tbl2:
        tbl.append(i)

    return tbl


# Send all the urls out to Google's SafeBrowsing API to check for
# malicious urls, and return the number found
def num_malicious_urls(urls):
    key = 'AIzaSyAAPunMDPhArqLnE_zH9ZK91VDGWxka8K8'
    lookup_url = 'https://safebrowsing.googleapis.com/v4/threatMatches:find?key={}'.format(key)

    url_list = ''
    for url in urls:
        url_list += '{{\"url\": \"{}\"}},\n'.format(url)

    payload = '{{\"client\" : \
               {{\"clientId\" : \"csci455\", \"clientVersion\" : \"0.0.1\"}}, \
               \"threatInfo\" : \
               {{\"threatTypes\" : [\"MALWARE\",\"SOCIAL_ENGINEERING\",\"UNWANTED_SOFTWARE\",\"MALICIOUS_BINARY\"], \
                \"platformTypes\" : [\"ANY_PLATFORM\"], \
                \"threatEntryTypes\" : [\"URL\"], \
                \"threatEntries\": [ {} ] \
                }} \
                }}'.format(url_list)
    r = requests.post(lookup_url, data=payload)
    if r.status_code == 200 and len(r.json()) > 0:
        return len(r.json()['matches'])
    return 0


def update_dow_ratios(weekday):
    dow_ratios[weekday] += 1
