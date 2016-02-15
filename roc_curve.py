
## ROC Curve



# ROC CURVES for 3 models
import sklearn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import seaborn as sns; 





## Get the data in shape. Train - Validation - Test

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




## MODEL 2: Logistic Regression with L1 Regularization


acclist = []
Cs = [0.2,0.4,0.6,0.8,1,5,10,20,100,1000]
for C in Cs:
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    #clf_l2_LR = LogisticRegression(C=1, penalty='l2', tol=0.01)
    clf_l1_LR.fit(X_train2, y_train2)

    base = np.repeat("DEM",len(y_valid))
    tru = sum(clf_l1_LR.predict(X_valid) == y_valid)
    #trul2 = sum(clf_l2_LR.predict(X_test) == y_test)


    trubase = sum(base == y_valid)
    tot = len(y_valid)
    acc = float(tru)/float(tot)
    acclist.append(acc)

C= Cs[acclist.index(max(acclist))]

print C


clf_l1_LRtest = LogisticRegression(C=C, penalty='l1', tol=0.01)

clf_l1_LRtest.fit(X_train, y_train)

y_pred_l1_LR = clf_l1_LRtest.predict_proba(X_test)[:, 1]
y_pred = ["DEM" if i < 0.37 else "REP" for i in y_pred_l1_LR]

fpr_l1LR, tpr_l1LR, _ = roc_curve(y_test, y_pred_l1_LR, pos_label='REP')









### Get the Confusion Matrix

from sklearn.metrics import confusion_matrix



mat = confusion_matrix(y_test, y_pred)
sns.set()


sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=['DEM','REP'], yticklabels=['DEM','REP'])
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()






