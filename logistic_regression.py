import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, classification_report
from scipy import sparse
from scipy.sparse import coo_matrix, vstack
import pandas as pd
import re
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

Xarr = np.load("/Users/tolga/Insight/Xarrfile.npy")
polparty = np.load("/Users/tolga/Insight/Xpartyfile.npy")
rows = Xarr.shape[0]
row_index = range(rows)


X_trainInd, X_testInd, y_train, y_test = train_test_split(row_index, polparty, test_size=0.10, random_state=42)


X_train = Xarr[X_trainInd]
X_test = Xarr[X_testInd]

X_trainInd2, X_validInd, y_train2, y_valid =  train_test_split(X_trainInd, y_train, test_size=0.10, random_state=42)

X_train2 = Xarr[X_trainInd2]
X_valid = Xarr[X_validInd]


C = 0.8

clf_l1_LRtest = LogisticRegression(C=C, penalty='l1', tol=0.01)

clf_l1_LRtest.fit(X_train, y_train)

y_pred_l1_LR = clf_l1_LRtest.predict_proba(X_test)[:, 1]

y_pred = ["DEM" if i < 0.37 else "REP" for i in y_pred_l1_LR]





df4 = pd.DataFrame.from_csv("/Users/tolga/Insight/ProjectDataFrame.csv")
df5 = df4.drop_duplicates(['TweetPre'])
df5 = df5.reset_index(drop=True)

ind = [False if pd.isnull(lst)  else True for lst in df5.TweetPre]
df6 = df5[ind]



features = []
for feature in df6.Features:
	try:
		features.append(feature[1:-1].split(' ,'))
	except: 
		features.append(feature)

df6['feature'] = features	


ind = [False if pd.isnull(lst)  else True for lst in df6.feature]
df7 = df6[ind]


allfeatures = ''
for feature in df7.feature:
	 try:
	 	allfeatures = allfeatures + ', ' + feature[0]
	 except:
	 	allfeatures = allfeatures + ''


allfeaturelist = re.split(r',\s*(?=[^)]*(?:\(|$))', allfeatures)

allfeatureset = set(allfeaturelist)
featurelist = list(allfeatureset)



print "df6 Pol Dist"
print df6.polparty.describe()

demwords = df7.TweetPre[df6.polparty=='DEM']
repwords = df7.TweetPre[df6.polparty=='REP']

demw = ''
for words in demwords:
    demw = demw + ', ' + words[1:-1]


text = re.sub('[^a-z\ \']+', " ", demw.lower())
demwordlist = list(text.split())


democratwords = set(demwordlist)


repw = ''
for words in repwords:
    repw = repw + ', ' + words[1:-1]


textrep = re.sub('[^a-z\ \']+', " ", repw.lower())
repwordlist = list(textrep.split())

republicanwords = set(repwordlist)

wordlistmerg = set(democratwords).union(republicanwords)


wordlistmerg = list(wordlistmerg)

count_onlydem = Counter()
count_onlyrep = Counter()



wordlistrep, wordfreqrep = zip(*Counter(repwordlist).most_common())
wordlistdem, wordfreqdem = zip(*Counter(demwordlist).most_common())











#DemHashZipped = zip(hashListDem,hashFreqDem)
#RepHashZipped = zip(hashListRep,hashFreqRep)

#draw_bar([DemHashZipped[index] for index in indexlst],'W3DemHash_freq')   
#draw_bar(RepHashZipped[2:15],'W3RepHash_freq')








import matplotlib.pyplot as plt 
import seaborn as sns; 
sns.set()

index = np.arange(len(wordlistrep[:40]))


plt.bar(index, wordfreqrep[:40],
                 alpha=1,
                 color='r',
                 label='Republican Words')
plt.xticks(index + 0.35, wordlistrep[:40],rotation=30)


sns.plt.show()





import vincent
from vincent import AxisProperties, PropertySet, ValueRef




DemZipped = zip(wordlistdem,wordfreqdem)
RepZipped = zip(wordlistrep,wordfreqrep)





def draw_bar(wordtuple,name,col):
    word_freq = wordtuple
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    #rotate x axis labels
    ax = AxisProperties(
         labels = PropertySet(angle=ValueRef(value=340)))
    bar.axes[0].properties = ax
    bar.color(brew=col)
    bar.to_json(name, html_out=True, html_path=name+'.html')




draw_bar(DemZipped[:20],"DemWordFreq","blue")
draw_bar(RepZipped[:20],"RepWordFreq","red")


coeftuple = zip(featurelist,clf_l1_LRtest.coef_[0])
tupsorted = sorted(coeftuple, key=lambda coef: coef[1]) 

draw_bar(tupsorted[-30:],'RepBest40Features',"red")
draw_bar(tupsorted[:30],'DemBest40Features',"blue")


