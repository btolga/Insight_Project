
## Import Packages


import tweepy
from tweepy import OAuthHandler
import operator 
from collections import Counter
from nltk import bigrams 
import nltk
import operator 
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import string
from numpy.random import normal
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random 
import json
from nltk.stem import WordNetLemmatizer


## Create DataFrame


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
myre = re.compile(u'('
    u'\ud83c[\udf00-\udfff]|'
    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
    u'[\u2600-\u26FF\u2700-\u27BF])+', 
    re.UNICODE)    
    
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via','RT','u','you','he','she','it','i','amp','&','he\'s','she\'s','it\'s','i\'m',
'bernie','sanders','bern','hillary','clinton','ted','cruz','donald','trump','don\'t','I\'m','would','could','can','will','we\re','you\'re']
numbers = [r'(?:(?:\d+,?)+(?:\.?\d+)?)']
htm = ['http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+']

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()



def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        tokens = [lmtzr.lemmatize(token) for token in tokens]


    return tokens


def tweetsasresults(filename,curr_class):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweet['pol'] = curr_class
            results.append(tweet)
    return results



resultsDem1 = tweetsasresults('/Users/tolga/Bernie4.json','DEM')
resultsRep1 = tweetsasresults('/Users/tolga/Trump4.json','REP')
resultsRep2 = tweetsasresults('/Users/tolga/Trump5.json','REP')
resultsRep3 = tweetsasresults('/Users/tolga/Trump8.json','REP')
resultsRep4 = tweetsasresults('/Users/tolga/Trump7.json','REP')


results = []
results.append(resultsDem1)
results.extend(resultsRep1)
results.extend(resultsRep2)
results.extend(resultsRep3)
results.extend(resultsRep4)


def toDataFrame(tweets):

    Data = pd.DataFrame()

    Data['tweetID'] = [tweet['id'] if 'id' in tweet else np.nan  for tweet in tweets]
    Data['tweetText'] = [tweet['text'].encode('ascii', 'ignore') if 'text' in tweet else np.nan  for tweet in tweets]
    Data['tweetRetweetCt'] = [tweet['retweet_count'] if 'retweet_count' in tweet else np.nan  for tweet in tweets]
    Data['tweetFavoriteCt'] = [tweet['favorite_count'] if 'favorite_count' in tweet else np.nan  for tweet in tweets]
    Data['tweetSource'] = [tweet['source'].encode('utf-8').decode('unicode_escape') if 'source' in tweet else np.nan  for tweet in tweets]
    Data['tweetCreated'] = [tweet['created_at'].encode('utf-8').decode('unicode_escape') if 'created_at' in tweet else np.nan  for tweet in tweets]

    Data['userID'] = [tweet['user']['id'] if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userScreen'] = [tweet['user']['screen_name'].encode('utf-8').decode('unicode_escape') if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userName'] = [tweet['user']['name'].encode('utf-8') if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userCreateDt'] = [tweet['user']['created_at'] if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userDesc'] = [tweet['user']['description'] if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userFollowerCt'] = [tweet['user']['followers_count'] if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userFriendsCt'] = [tweet['user']['friends_count'] if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userLocation'] = [tweet['user']['location'] if 'user' in tweet else np.nan  for tweet in tweets]
    Data['userTimezone'] = [tweet['user']['time_zone'] if 'user' in tweet else np.nan  for tweet in tweets]

    Data['polparty'] = [tweet['pol'] if 'user' in tweet else np.nan  for tweet in tweets]
    return Data

dfDem1 = toDataFrame(resultsDem1)
dfRep1 = toDataFrame(resultsRep1)
dfRep2 = toDataFrame(resultsRep2)
dfRep3 = toDataFrame(resultsRep3)
dfRep4 = toDataFrame(resultsRep4)


rows = random.sample(dfDem1.index, 90000)
dfDem1drop = dfDem1.ix[rows]
df_Dem1nondrop = dfDem1.drop(rows)



frames = [dfDem1drop, dfRep1, dfRep2, dfRep3, dfRep4]
df = pd.DataFrame()
df = pd.concat(frames)


df2 = df.dropna(subset=['tweetText'], how='all')
df3 = df2.drop_duplicates(['userID'])



minlength = 2
tweetpre = []
for tweet in df3.tweetText:
    terms_only2 = [term.encode('unicode-escape') for term in preprocess(tweet) 
                   if term.lower() not in stop and
                   not term.lower().startswith(('#', '@')) and
                   not term.lower().startswith(('htt', '\u')) and 
                   term.lower() not in [ r'(?:(?:\d+,?)+(?:\.?\d+)?)'] and
                   len(term) > minlength]
    #terms_only2.extend(bigrams(terms_only2))
    tweetpre.append(terms_only2)

df3.loc[:,'TweetPre'] = tweetpre


ind = [False if lst ==[] else True for lst in df3.TweetPre]
df4 = df3[ind]





df4 = df4.reset_index(drop=True)


print df4.TweetPre[:40]

print df4.polparty.describe()


tweetbigrams = []
for tweet in df4.TweetPre:
	tweetbigrams.append(list(bigrams(tweet)))



print tweetbigrams[:3]


df4['Bigrams'] = tweetbigrams

df4['Features'] = df4.TweetPre + df4.Bigrams


df4.to_csv('/Users/tolga/Insight/ProjectDataFrame.csv', encoding ='utf-8')
