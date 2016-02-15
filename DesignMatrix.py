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


'''
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
'''


allfeatures = ''
for feature in df7.feature:
	 try:
	 	allfeatures = allfeatures + ', ' + feature[0]
	 except:
	 	allfeatures = allfeatures + ''


allfeaturelist = re.split(r',\s*(?=[^)]*(?:\(|$))', allfeatures)

allfeatureset = set(allfeaturelist)
featurelist = list(allfeatureset)



'''
allfeatures = []
for feature in df6.feature:
	try:
		allfeatures.append(feature[0])
	except:
		allfeatures.append(feature)

allfeaturestr = ''
for feature in allfeatures:
	try:
		allfeaturestr = allfeaturestr + feature
	except:
		allfeaturestr = allfeaturestr

print allfeaturestr[:10]

allfeaturelist = allfeaturestr.split(' ,')


allfeatureset = set(allfeaturelist)
featurelist = list(allfeatureset)

print featurelist[0]

'''


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


'''

X = []    
i = 1
for tweet in df7.feature[7000:12000]:

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

print Xarr.shape
print len(df7.polparty)



np.save("Xarrfile2", Xarr)


np.save("Xpartyfile2", df7.polparty)


del(Xarr)
X = []    
i = 1
for tweet in df7.feature[12000:]:

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

print Xarr.shape
print len(df7.polparty)



np.save("Xarrfile3", Xarr)


np.save("Xpartyfile3", df7.polparty)
'''
