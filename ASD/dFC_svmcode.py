# python code for dynamical functional connectivity matrices
%matplotlib
import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# this function returns all functional matrices for each subject
def dFC_matrices(folder):

    featureMatrix = []

    subjects_dFCList = []    # a list for dFC matrices for all subjects

    filelist = os.listdir(folder)

    for i in range(len(filelist)):

        filename = folder + '/' + filelist[i]

        #Each row is a region, each column is a time point
        brainData = np.loadtxt(open(filename, "rb"), delimiter=",")
        np.shape(brainData)

        windLength, stepSize = 30, 30
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

        featureMatrix.append(single1D_Vec)

        """
        single1D_Vec = []
        for x in subject_triList:
            single1D_Vec.append(np.ravel(x))

        featureMatrix.append(single1D_Vec)
        """
        ####################################################

        subjects_dFCList.append(subject_triList)

    return featureMatrix


# this code returns the 1D of upper triangular for all subjects
def subjects_triList():

    allSubjects_dFC_matrices_ASD = dFC_matrices('ASD')
    allSubjects_dFC_matrices_control = dFC_matrices('control')

    X =  np.vstack((allSubjects_dFC_matrices_control,allSubjects_dFC_matrices_ASD))

    return X


X = subjects_triList()
y = pd.read_csv("ASD_labels_SDSU.csv")
y = np.ravel(y)
#print(X)
#print(y)
print(np.shape(X))
print(np.shape(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
########################################################## SVM Code

clf = svm.SVC(kernel = "linear", probability = True, C = 00.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

########################################################## Logistic regression code
"""
clf = LogisticRegression(penalty='l2', C=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""
######################### Model evaluation ###################
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
print("Accuracy", metrics.accuracy_score(y_test, y_pred))

# ROC Curve and AUC
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label = 2)
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

lw = 2

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
