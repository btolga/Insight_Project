## THIS CELL CREATES WORD LISTS FOR BERNIE AND TRUMP
import tweepy
from tweepy import OAuthHandler
import operator 
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from numpy.random import normal
from nltk import bigrams 
import pandas as pd
import numpy as np

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
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 


def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        tokens = [lmtzr.lemmatize(token) for token in tokens]


    return tokens


punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via','RT','u','you','he','she','it','i','amp','&','bernie','sanders','bern','hillary','clinton','ted','cruz','donald','trump']
numbers = [r'(?:(?:\d+,?)+(?:\.?\d+)?)']
htm = ['http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+']




def wordlistfun(filename):
    minlength = 2
    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
    wordlist = []
    wordfreq = []
    hashlist = []
    hashfreq = []

    with open(filename, 'r') as f:
        count_all = Counter()
        count_hash = Counter()
        count_only = Counter()
        count_bi = Counter()
        count_only2 = Counter()
        count_bigramonly = Counter()
        count_bigramstop = Counter()
        for line in f:
            try:
                tweet = json.loads(line)
                # Create a list with all the terms
                terms_stop = [term for term in preprocess(tweet['text']) if term.lower() not in stop]        # Update the counter
                terms_hash = [term for term in preprocess(tweet['text']) 
                        if term.lower().startswith('#')]
                terms_only = [term for term in preprocess(tweet['text']) 
                        if term.lower() not in stop and
                        not term.lower().startswith(('#', '@'))] 
                        # mind the ((double brackets))
                        # startswith() takes a tuple (not a list) if # we pass a list of inputs
                terms_only2 = [term.encode('unicode-escape') for term in preprocess(tweet['text']) 
                        if term.lower() not in stop and
                        not term.lower().startswith(('#', '@')) and
                        not term.lower().startswith(('htt', '\u')) and 
                        term.lower() not in [ r'(?:(?:\d+,?)+(?:\.?\d+)?)'] and
                        len(term) > minlength]

                terms_bigramstop = bigrams(terms_stop)
                terms_bigramonly = bigrams(terms_only2)



                count_all.update(terms_stop)
                count_hash.update(terms_hash)
                count_only.update(terms_only)
                count_only2.update(terms_only2)

                count_bigramonly.update(terms_bigramonly)
                count_bigramstop.update(terms_bigramstop)
            except:
                pass

        wordlist, wordfreq = zip(*count_only2.most_common())
        hashlist, hashfreq = zip(*count_hash.most_common())
    return wordlist, wordfreq, hashlist, hashfreq


wordListDem1 , wordFreqDem1, hashListDem1, hashFreqDem1 = wordlistfun('/Users/tolga/Bernie4.json')
wordListRep1 , wordFreqRep1, hashListRep1, hashFreqRep1 = wordlistfun('/Users/tolga/Trump4.json')
wordListRep2 , wordFreqRep2, hashListRep2, hashFreqRep2 = wordlistfun('/Users/tolga/Trump5.json')
wordListRep3 , wordFreqRep3, hashListRep3, hashFreqRep3 = wordlistfun('/Users/tolga/Trump7.json')
wordListRep4 , wordFreqRep4, hashListRep4, hashFreqRep4 = wordlistfun('/Users/tolga/Trump8.json')



hashListRep = hashListRep1 + hashListRep2 + hashListRep3 + hashListRep4
hashFreqRep = hashFreqRep1 + hashFreqRep2 + hashFreqRep3 + hashFreqRep4



DemHashZipped = zip(hashListDem1,hashFreqDem1)
RepHashZipped = zip(hashListRep,hashFreqRep)


import vincent
from vincent import AxisProperties, PropertySet, ValueRef






def draw_bar(wordtuple,name):
    word_freq = wordtuple
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    #rotate x axis labels
    ax = AxisProperties(
         labels = PropertySet(angle=ValueRef(value=340)))
    bar.axes[0].properties = ax
    bar.to_json(name, html_out=True, html_path=name+'.html')



draw_bar(DemHashZipped[17:32],'DemHash')   
draw_bar(RepHashZipped[15:30],'RepHash')




from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS

twitter_mask = imread('./Users/tolga/Insight/twitter_mask.png', flatten=True)


wordcloud = WordCloud(
                      font_path='/Users/tolga/Insight/cabin-sketch-v1.02/CabinSketch-Bold.ttf',
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=1800,
                      height=1400,
                      mask=twitter_mask
            ).generate(DemHashZipped)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('./Users/tolga/Insight/wordcloudDEMHash.png', dpi=300)
plt.show()
