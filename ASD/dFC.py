import numpy as np


# assume we have a data of 150 columns
# sliding window analysis
# window length = 50
# step size = 50

def dFC_matrices(filename):

    brainData = np.loadtxt(open(filename, "rb"), delimiter=",") #Each row is a region, each column is a time point
    np.shape(brainData)

    windLength, stepSize = 50, 50
    #stepSize = 50
    leftStile, rightStile = 0, windLength + 1
    #rightStile = windLength + 1
    dataParts = []
    dFC_matrices = []

    while rightStile <= len(A[0]):
        dataParts.append(brainData[0:,leftStile:rightStile])
        leftStile = leftStile + stepSize
        rightStile = rightStile + stepSize
  

    # the data size, window length and step size means we get three partition
    # we create three matrices for the partition
    #part_one = brainData[0:,0:51]
    #part_two = brainData[0:,51:101]
    #part_three = brainData[0:,101:151]

    for each in dataParts:
        dFC_matrices.append(np.corrcoef(each))

    # we create three functional connectivity matrices from the parts above
    #dFC_matrix_one = np.corrcoef(part_one)
    #dFC_matrix_two = np.corrcoef(part_two)
    #dFC_matrix_three = np.corrcoef(part_three)

    # put all dFC matrices in one list
    #dFC_matrices = [dFC_matrix_one,dFC_matrix_two,dFC_matrix_three]

    return dFC_matrices

# this function calculates dFC for all subjects in the study
def dFC_matrices_allSubject():
    subjects_dFCList = []    # a list for dFC matrices for all subjects
    for i in range(146):
        filenumber = 40000
        filename = str(filenumber).zfill(7) + ".csv"

        #filename = seudoFilename + ".csv"

        A = np.loadtxt(open(filename, "rb"), delimiter=",") #Each row is a region, each column is a time point
        np.shape(A)

        dFC_matrices_subjectN = dFC_matrices(filename)
        subjects_dFCList.append(dFC_matrices_subjectN)


        filenumber = filenumber + 1

    return subjects_dFCList


def allSubjects_triangular():
    triangularList = []
    allSubjects_dFC_matrices = dFC_matrices_allSubject()

    #print(a[np.triu_indices(5)])

    subjectN_triList = []
    for i in allSubjects_dFC_matrices: # i for three set of dFC for each subject
    	for j in i:     # j for a single dFC matrix in a subjects three set of dFC matrices
    		subjectN_triList.append(j[np.triu_indices()])

    	triangularList.append(subjectN_triList)

    return triangularList
