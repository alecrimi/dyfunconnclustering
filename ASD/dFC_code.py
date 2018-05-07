# python code for dynamical functional connectivity matrices
import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
#####################################################
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# this function returns all functional matrices for each subject
def dFC_matrices(folder):

    #subjects_dFCList = []    # a list for dFC matrices for all subjects
    featuresMatrix = []
    filelist = os.listdir(folder)

    for i in range(len(filelist)):

        filename = folder + '/' + filelist[i]

        #Each row is a region, each column is a time point
        brainData = np.loadtxt(open(filename, "rb"), delimiter=",")
      
        windLength, stepSize = 180, 50
        leftStile, rightStile = 0, windLength


        # the data size, window length and step size means we get three partition
        dataParts = []
        while rightStile <= len(brainData[0]):
            dataParts.append(brainData[0:,leftStile:rightStile])
            leftStile = leftStile + stepSize
            rightStile = rightStile + stepSize

        # creating correlation matrix for each data partition
        dFC_matrices = []
        for each in dataParts: 
            dFC_matrices.append(np.corrcoef(each)) 
        #return dFC_matrices

        #Add for loop here to reduce to tria
        subject_triList = []
        for j in dFC_matrices:
            subject_triList.append(j[np.triu_indices (len(dFC_matrices[0][0]),1)] )
       

        ######################################### combining all 1d vectors into 1d feature vector
        
        single1D_Vec = []
        for x in subject_triList:
            for element in x:
                single1D_Vec.append(element)
        
        featuresMatrix.append(single1D_Vec)

        ####################################################

       # subjects_dFCList.append(subject_triList)

    #return subject_triList
    #return subjects_dFCList
    return featuresMatrix


# this code returns the 1D of upper triangular for all subjects
def subjects_triList():

    allSubjects_dFC_matrices_ASD = dFC_matrices('ASD')
    r,c = np.shape(allSubjects_dFC_matrices_ASD)
   # print(np.shape(allSubjects_dFC_matrices_ASD))
    allSubjects_dFC_matrices_control = dFC_matrices('control')
    r2,c2 = np.shape(allSubjects_dFC_matrices_control)
    #print(np.shape(allSubjects_dFC_matrices_control))
    X =  np.vstack((allSubjects_dFC_matrices_control,allSubjects_dFC_matrices_ASD))
    y =  np.hstack((np.ones(r2),np.zeros(r)))
    return X,y


X,y = subjects_triList()
#y = pd.read_csv("ASD_labels_SDSU.csv")
 
#y = np.ravel(y)
#print(X)
#print(y)
#print(np.shape(X))
#print(np.shape(y))

loo = LeaveOneOut()

r = np.shape(y)
score= np.zeros(r)

count = 0

train_X=np.zeros((53,len(X[0,:])))

train_y = np.zeros(53)

test_X=[0]

test_y=[0]

for train_index, test_index in loo.split(X):

    for i in range(len(train_index)):

        train_X[i,:]=X[train_index[i]]

        train_y[i]=y[train_index[i]]

    test_X=X[test_index[0]]

    test_y=y[test_index[0]]

    clf = svm.SVC(kernel='linear', C=1, probability=True).fit(train_X, train_y)

    probs = clf.predict_proba([test_X])

    score[count] = probs[:,0]
		 
    count+=1

 

roc_x = []

roc_y = []

min_score = min(score)

max_score = max(score)

thr = np.linspace(min_score, max_score, 70)

FP=0

TP=0

P = sum(y[y==1])
N = len(y) - P

for (i,T) in enumerate(thr):
		for i in range(0, len(score)):
			if (score[i] > T):

				if (y[i]==1):
					TP = TP + 1
				if (y[i]==0):
					FP = FP + 1
		
		roc_x.append(FP/float(N))
		roc_y.append(TP/float(P))
		FP=0
		TP=0
     
 
roc_auc= auc( roc_y,roc_x)
print(roc_auc)
##############################################################################

#Plot of a ROC curve for a specific class

lw = 2

plt.plot(roc_y, roc_x, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
