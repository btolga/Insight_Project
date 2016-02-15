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

##LOGISTIC REG MODEL WITH L1 Reg Plots etc

Xarr = np.load("/Users/tolga/Insight/Xarrfile.npy")
polparty = np.load("/Users/tolga/Insight/Xpartyfile.npy")
print "Size of the 1st Matrix"
print Xarr.shape
X_sp1 = sparse.coo_matrix(Xarr)

'''

Xarr2 = np.load("/Users/tolga/Insight/Xarrfile.npy")
polparty2 = np.load("/Users/tolga/Insight/Xpartyfile.npy")
X_sp2 = sparse.coo_matrix(Xarr2)

print "Size of the 2nd Matrix"
print Xarr2.shape
del(Xarr2)



Xarr3 = np.load("/Users/tolga/Insight/Xarrfile.npy")
polparty3 = np.load("/Users/tolga/Insight/Xpartyfile.npy")
X_sp3 = sparse.coo_matrix(Xarr3)

print "Size of the 3rd Matrix"
print Xarr3.shape
del(Xarr3)



X_sp = vstack([X_sp1,X_sp2, X_sp3]).toarray()

print X_sp.shape
print polparty.shape
'''


rows = Xarr.shape[0]
row_index = range(rows)







X_trainInd, X_testInd, y_train, y_test = train_test_split(row_index, polparty, test_size=0.10, random_state=42)


X_train = Xarr[X_trainInd]
X_test = Xarr[X_testInd]


rows2 = X_train.shape[0]
row2_index = range(rows2)

X_trainInd2, X_validInd, y_train2, y_valid =  train_test_split(row2_index, y_train, test_size=0.10, random_state=42)

X_train2 = X_train[X_trainInd2]
X_valid = X_train[X_validInd]

C = 5

clf_l1_LRtest = LogisticRegression(C=C, penalty='l1', tol=0.01)

clf_l1_LRtest.fit(X_train, y_train)

y_pred_l1_LR = clf_l1_LRtest.predict_proba(X_test)[:, 1]
y_pred = ["DEM" if i < 0.45 else "REP" for i in y_pred_l1_LR]

fpr_l1LR, tpr_l1LR, _ = roc_curve(y_test, y_pred_l1_LR, pos_label='REP')
