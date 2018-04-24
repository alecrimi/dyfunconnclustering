# python code for dynamical functional connectivity matrices

import numpy as np


# assume we have a data of 150 columns
# sliding window analysis
# window length = 50
# step size = 50

# this function returns all functional matrices for a single subject
def dFC_matrices(filename):

    #Each row is a region, each column is a time point
    brainData = np.loadtxt(open(filename, "rb"), delimiter=",")
    np.shape(brainData)

    windLength, stepSize = 50, 50
    leftStile, rightStile = 0, windLength

    # we define two matrices: one to hold all the partitions and the other to
    #                 hold all functional matrices for a subject
    dataParts = []
    dFC_matrices = []

    # the data size, window length and step size means we get three partition
    while rightStile <= len(brainData[0]):
        dataParts.append(brainData[0:,leftStile:rightStile])
        leftStile = leftStile + stepSize
        rightStile = rightStile + stepSize

    # creating correlation matrix for each data partition
    for each in dataParts:
        dFC_matrices.append(np.corrcoef(each))

    return dFC_matrices


# this function calculates dFC for all subjects in the study
def dFC_matrices_allSubject():
    subjects_dFCList = []    # a list for dFC matrices for all subjects
    filenumber = 40000
    for i in range(146):

        filename = str(filenumber).zfill(7) + ".csv"
        dFC_matrices_subjectN = dFC_matrices(filename)
        subjects_dFCList.append(dFC_matrices_subjectN)

        filenumber = filenumber + 1

    return subjects_dFCList



# this code returns the 1D of upper triangular for all subjects
def allSubjects_triangular():
    triangularList = []
    allSubjects_dFC_matrices = dFC_matrices_allSubject()

    subjectN_triList = []
    for i in allSubjects_dFC_matrices: # i for three set of dFC for each subject
    	for j in i:     # j for a single dFC matrix in a subjects three set of dFC matrices
    		subjectN_triList.append(j[np.triu_indices(len(allSubjects_dFC_matrices[0][0]))])

    	triangularList.append(subjectN_triList)

    return triangularList
