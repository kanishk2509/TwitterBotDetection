import tweepy
import datetime as dt
import requests
import re
from GetTweetProperties import get_tweet_properties, get_tweet_semantics

dow_ratios = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

'''
Step 3
Calculating Twitter User Account Properties Component
'''


def get_data(user_id, api):
    tbl = []
    try:
        tbl = mine_data(user_id, api)
        return tbl
    except tweepy.TweepError as e:
        print(e)
        return tbl


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

    tbl2 = get_tweet_properties(user_id, api, user)

    for i in tbl2:
        tbl.append(i)

    std_dev_friends, \
        std_dev_followers, unique_urls_ratio, \
        tweet_url_similarity, user_desc_len, user_desc_sentiment, \
        special_char_count, tweet_count \
        = get_tweet_semantics(user_id, api)

    tbl.append(std_dev_friends)
    tbl.append(std_dev_followers)
    tbl.append(unique_urls_ratio)
    tbl.append(tweet_url_similarity)
    tbl.append(user_desc_len)
    tbl.append(user_desc_sentiment)
    tbl.append(special_char_count)
    tbl.append(tweet_count)
    tbl.append(bot_flag)

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
