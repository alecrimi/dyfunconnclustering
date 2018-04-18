import numpy as np

filename = "0040000.csv"

A = np.loadtxt(open(filename, "rb"), delimiter=",")
np.shape(A)

#Each row is a region, each column is a time point
