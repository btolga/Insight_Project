## Import Pacakges

import pandas as pd
import numpy as np
import re 
import string
from nltk import bigrams 


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

demwords = df6.feature[df6.polparty=='DEM']
repwords = df6.feature[df6.polparty=='REP']



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





d = {}
for i in range(len(featurelist)):
    d[featurelist[i]] = i



import numpy as np

X = []    
i = 1
for tweet in df7.feature:
    temp = np.zeros(len(featurelist))
    string = tweet[0]
    r = re.compile(r'(?:[^,(]|\([^)]*\))+')
    txtlist = r.findall(string)
    for word in txtlist:
        if word[0] != ' ':
          try:
            temp[d[word]] = 1
          except:
            pass
        else:
          try:
            temp[d[word[1:]]] = 1
          except:
            pass
    X.append(temp)
    i = i+1



Xarr = np.array(X)


np.save("Xarrfile", Xarr)


np.save("Xpartyfile", df7.polparty)

del(Xarr)


