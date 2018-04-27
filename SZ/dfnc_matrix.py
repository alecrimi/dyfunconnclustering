import numpy as np
import os

#this function cut each subject in three subparts

def window_per_people(subject):

	line_col=np.shape(subject)
	line=line_col[0] #number of line
	col=line_col[1] #number of column

	shape =(96,50) #number of line and column 
	window_one= np.zeros(shape)
	window_two= np.zeros(shape)
	window_three= np.zeros(shape)
		
	for i in range(0,line):
		line_btd=subject[i] # we take each line 
		for j in range(0,col):
			if (j<50 and j>=0):
				w=line_btd[j]
				window_one[i][j]=w # put in the first window
			elif (j<100 and j>=50): 
				w=line_btd[j]
				window_two[i][j-50]=w # put in the second window
			elif (j<150 and j>=100):
				w=line_btd[j]
				window_three[i][j-100]=w # put in the third window
				
	set_window=[window_one, window_two, window_three]
	return set_window

	
# this function compute the correlation matrix

def static_connectivity_matrix(subject):
	corr= np.corrcoef(subject)	
	return corr 
	
#flat the connectivity matrix in 1D
def features(correlation_matrix_static):

	corr= correlation_matrix_static
	vec_1D_per_subject_static=[0]
		
	#we take triangular superior of the correlation matrix 
	triangular_sup = np.triu(corr,k=0)
	
	#transformation of the triangular matrix in 1D vector  
	vec_1D_per_subject_static=vec_1D_per_subject_static+list(triangular_sup[np.triu_indices(96)])
	
	del vec_1D_per_subject_static[0]# delete the initial value 
		
	return vec_1D_per_subject_static
	
def dynamical_connectivity(win_set):
	set_window= win_set
	
	dynamic_mat_list=[]
	
	for window in range(len(set_window)):
		
		# we compute the static connectivity per window which makes the dynamic		
		corr = static_connectivity_matrix(set_window[window]) 
			
		# take upper part of the correlation matrix
		triangular_sup = np.triu(corr,k=0)
			
		dynamic_mat_list.append(triangular_sup)
		
	return dynamic_mat_list
	

if __name__=="__main__":
	sz_one_dim=[]
	sz_dynamic_matrix=[]
	
	control_one_dim=[]
	control_dynamic_matrix= []
	
	#data schizophrenia
	filename_sz = os.listdir("SZ")
	for filename in filename_sz: 
		subject = np.loadtxt(open("SZ/"+filename, "rb"), delimiter=",")
		static_CM = static_connectivity_matrix(subject)
		sz_one_dim.append(features(static_CM))
		win_set = window_per_people(subject)
		sz_dynamic_matrix.append(dynamical_connectivity(win_set))
	
		
        #data control healthy
	filename_control = os.listdir("CONTROL")
	for filename in filename_control:
		subject = np.loadtxt(open("CONTROL/"+filename, "rb"), delimiter=",")
		static_CM = static_connectivity_matrix(subject)
		control_one_dim.append(features(static_CM))
		win_set = window_per_people(subject)
		control_dynamic_matrix.append(dynamical_connectivity(win_set))
	
