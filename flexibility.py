###################### Flexibility index ##################################
# 
#   This script calculates the flexibility coefficient of S.
#   The flexibility of each node corresponds to the number of times that
#   it changes module allegiance, normalized by the total possible number
#   of changes. In temporal networks, we consider changes possible only
#   between adjacent time points. In multislice/categorical networks,
#   module allegiance changes are possible between any pairs of slices.
#
#   This metric was defined by Bassett, Danielle S., et al. 
#   in "Dynamic reconfiguration of human brain networks during learning." 
#   Proceedings of the National Academy of Sciences (2011).
#
#   This script expects to load a matrix S, pxn  of community assignments 
#   where p is the number of slices/layers and n the number of nodes
#
# It is assumed that such ordening is on a CSV file

import csv
import numpy as np

nettype='temp' # For only flexibility on adjacent slices
#nettype='cat' # consider all pairs of slices

####### LOAD DATA
with open('data.csv') as csvfile:
    S = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
S = np.asarray(S)

[numSlices, numNodes] = np.shape(S);

######## CALCULATE FLEXIBILITY
# Pre-allocate totalChanges
totalChanges = np.zeros(numNodes)

if (nettype == 'temp'):
    possibleChanges = numSlices-1 # only consider adjacent slices, except the last one as there is nothing after
    for t in range(1,numSlices):
        totalChanges = totalChanges + (  1*np.not_equal( S[t,:], S[t-1,:] )  );
 
elif (nettype == 'cat'):
    possibleChanges = numSlices*(numSlices-1); 
    for s in range(1,numSlices):
         otherSlices = range(1,numSlices+1) 
         otherSlices.remove(s) # all slices but the current one 
         totalChanges = totalChanges + sum(  1*np.not_equal(  np.repeat(S(s,:), (numSlices-1)),  S(otherSlices,:) )  )
 

######### CALCULATE OUTPUT  
F = totalChanges/possibleChanges
print(F) 
