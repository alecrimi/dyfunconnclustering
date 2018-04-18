import numpy as np

filename = "0040000.csv"

# Load file
A = np.loadtxt(open(filename, "rb"), delimiter=",") #Each row is a region, each column is a time point
np.shape(A)

# Compute functional connectivity
C = np.corrcoef(A) #This is the static functional connectivity, to get the dynamical you have to select only few columns
