import numpy as np
import tweepy
from nltk import PorterStemmer, word_tokenize, re
from nltk.corpus import stopwords
from collections import Counter
from CosineSentenceSimilarity import get_avg_cosine_similarity
from AverageSentenceSentiment import get_avg_sentiment, get_avg_sentiment_single
from StdDeviationFollowers import compute_std_deviation_followers
from StdDeviationFriends import compute_std_deviation_friends

'''
Returns 7 tweet properties listed below

    1. Average tweet per day
    2. Hashtags ratio
    3. User mentions ratio
    4. URL ratio
    5. Cosine Similarity
    6. Average Sentiment
    7. Bot flag
    
'''


def get_tweet_properties(user_id, api, user):
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

                cos_sim = get_avg_cosine_similarity(sent_array)
                avg_sentiment = get_avg_sentiment(sent_array)

            else:
                cos_sim, avg_sentiment = 0.0, 0.0

            '''if len(urls) > 0:
                        mal_urls_ratio = num_malicious_urls(urls) / len(urls)
                        print("mal_urls_ratio: ", mal_urls_ratio)'''

            tbl.append(cos_sim)
            tbl.append(avg_sentiment)

            return tbl

        except tweepy.TweepError as e:
            print(e)
            return []

    else:
        print("Protected Account: {}".format(user_id))
        return []


'''
Returns 8 tweet semantics listed below

    1. Standard deviation of user's friends
    2. Standard deviation of user's followers
    3. Average number of "unique" urls in all the tweets, since bots tend to share similar urls over time.
    4. Similarity in URLs, since bots tend to promote a single domain to increase user traffic on their website
    5. Length of user description
    6. Sentiment of user's description
    7. Number of special character in user's description 
    8. Total number of tweets
    
'''


def get_tweet_semantics(user_id, api):
    user = api.get_user(user_id)
    tbl = []
    urls = []

    # Calculating (1, 2)
    # 1
    std_dev_friends = compute_std_deviation_friends(user_id, api)
    # 2
    std_dev_followers = compute_std_deviation_followers(user_id, api)
    print("std_dev_friends : ", std_dev_friends)
    print("std_dev_followers : ", std_dev_followers)
    tbl.append(0)

    if std_dev_friends > 0:
        tbl.append(std_dev_friends)
    else:
        tbl.append('nan')

    if std_dev_followers > 0:
        tbl.append(std_dev_followers)
    else:
        tbl.append('nan')

    if not user.protected:
        try:
            for tweet in tweepy.Cursor(api.user_timeline, id=user_id, tweet_mode='extended').items(1000):
                if len(tweet.entities['urls']) > 0:
                    for url in tweet.entities['urls']:
                        urls.append(url['expanded_url'])
            # Calculating (3, 4)
            if len(urls) > 0:
                # 3
                unique_urls_ratio = len(Counter(urls).keys()) / len(urls)
                # 4
                tweet_url_similarity = get_avg_cosine_similarity(urls, 'URLs')
                tbl.append(unique_urls_ratio)
                tbl.append(tweet_url_similarity)
            else:
                unique_urls_ratio = 'nan'
                tweet_url_similarity = 'nan'
                tbl.append(unique_urls_ratio)
                tbl.append(tweet_url_similarity)
                print('unique_urls_ratio : ', unique_urls_ratio)
                print('tweet_url_similarity :', tweet_url_similarity)
        except tweepy.TweepError as e:
            print("Some error occurred! ", e)
            return []

        # Calculating (5, 6, 7, 8)
        description = user.description
        # 5
        user_desc_len = len(description)
        # 6
        user_desc_sentiment = get_avg_sentiment_single(description, 'user description')
        # 7
        special_char_count = len(re.sub('[\w]+', '', description))
        # 8
        tweet_count = user.statuses_count

        tbl.append(user_desc_len)
        tbl.append(user_desc_sentiment)
        tbl.append(special_char_count)
        tbl.append(tweet_count)

        return tbl

    else:
        print("Protected Account: {}".format(user_id))
        return []
