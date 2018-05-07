import numpy as np
import os
import math as mt
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import auc

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

	
# this function computes the correlation matrix

def static_connectivity_matrix(subject):
	corr= np.corrcoef(subject)	
	return corr 
	
#this function flats the static connectivity matrix in 1D
def features(correlation_matrix_static):

	corr= correlation_matrix_static
	vec_1D_per_subject_static=[0]
		
	#we take triangular superior of the correlation matrix 
	triangular_sup = np.triu(corr,k=0)
	
	#transformation of the triangular matrix in 1D vector  
	vec_1D_per_subject_static=vec_1D_per_subject_static+list(triangular_sup[np.triu_indices(96)])
	
	del vec_1D_per_subject_static[0]# delete the initial value
	 
	#the function nan_to_num replace nan with zero and inf with large finite numbers. 
	vec_1D_per_subject_static=np.nan_to_num(vec_1D_per_subject_static)
	
	return vec_1D_per_subject_static

#this function flats the dynamical connectivity matrix in 1D
def dynamical_connectivity(win_set):
	set_window= win_set
	
	dynamic_mat_list=[]
	
	for window in range(len(set_window)):
		
		# we compute the static connectivity per window which makes the dynamic		
		corr = static_connectivity_matrix(set_window[window]) 
					
		# the function features flats the matrix triangular superior in dynamical connect...
		a=list(features(corr))
		dynamic_mat_list=dynamic_mat_list+a
		
	return dynamic_mat_list
	
# Cross validation
def Leave_one_out(X,y):
	
	loo = LeaveOneOut()
	n=143
	X_train=[[]*n for x in xrange(n)]
	X_test=[]
	y_train=[]
	y_test=[]
	count=0
	score= np.zeros(144)
	
	for train_index, test_index in loo.split(X):

		for i in train_index:
 		
			if i>test_index[0]:
				X_train[i-1]=X[train_index[i-1]]
				y_train.append(y[i-1])
			else:
				X_train[i]=X[train_index[i]]
				y_train.append(y[i])

		X_test=X[test_index[0]]
		y_test=y[test_index[0]]
		
		clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)

		probs = clf.predict_proba(X_test.reshape(1,-1))
		score[count] = probs[:,0]
		count+=1
		
		#clear the lists
	
		X_train=[[]*n for x in xrange(n)]
		X_test=[]
		y_train=[]
		y_test=[]
	
	return score

	
def Alessandro_ROC(y_preditc_proba,y):
	score=y_preditc_proba
	roc_x = []
	roc_y = []
	min_score = min(score)
	max_score = max(score)
	thr = np.linspace(min_score, max_score, 30)
	FP=0
	TP=0
	P = sum(y)
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

	return  roc_x,roc_y

#Plotting of a ROC curve for a specific class
def plot_ROC(roc_x,roc_y,roc_name):

	
	list_x=roc_x
	list_y=roc_y
	roc_auc= auc(roc_x, roc_y)
	lw = 2
	plt.plot(list_x,list_y, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(str(roc_name)+' ROC')
	plt.legend(loc="lower right")
	plt.show()

#This function does the experiment with the static connectivity
def first_experiment_static(sz_one_dim, control_one_dim, y,roc_name):
	
	#matrix of all subject in schizophrenia and control healthy
	allSubjects_dFC_matrices_SZ=sz_one_dim
	allSubjects_dFC_matrices_control=control_one_dim
	
	# concatenation of the two matrix i.e the whole dataset
	X =  np.concatenate((allSubjects_dFC_matrices_SZ, allSubjects_dFC_matrices_control), axis=0)
        print(np.shape(X))
	y_preditc_proba=Leave_one_out(X,y)
	
	#y_preditc_proba=clf.predict_proba(xtrain)[:,1]
	
	roc_x,roc_y =Alessandro_ROC(y_preditc_proba,y)
	
	#plotting the ROC curve
	plot_ROC(roc_y,roc_x,roc_name)


# This function does the experiment with the dynamical connectivity
def second_experiment_dynamic(sz_dynamic_one_dim,control_dynamic_one_dim,y):

	roc_name = 'Dynamical Connectivity'
	first_experiment_static(sz_dynamic_one_dim,control_dynamic_one_dim,y,roc_name)


if __name__=="__main__":
	sz_one_dim=[]			
	sz_dynamic_one_dim=[]		
	
	control_one_dim=[]
	control_dynamic_one_dim= []
	
	#data schizophrenia
	filename_sz = os.listdir("SZ")
	for filename in filename_sz: 
		subject = np.loadtxt(open("SZ/"+filename, "rb"), delimiter=",")
		static_CM = static_connectivity_matrix(subject)
		sz_one_dim.append(features(static_CM))
		
		win_set = window_per_people(subject)
		sz_dynamic_one_dim.append(dynamical_connectivity(win_set))
	
		
        #data control healthy
	filename_control = os.listdir("CONTROL")
	for filename in filename_control:
		subject = np.loadtxt(open("CONTROL/"+filename, "rb"), delimiter=",")
		static_CM = static_connectivity_matrix(subject)
		control_one_dim.append(features(static_CM))
		
		win_set = window_per_people(subject)
		control_dynamic_one_dim.append(dynamical_connectivity(win_set))
	print np.shape(control_one_dim)

	#notation
	y=np.zeros(144)
	y[0:70]=1    #subject get the schizophrenia
	y[70:144]=0  #subject healthy
		
	
	#the two experiments
	roc_name = 'Static Connectivity'
	first_experiment_static(sz_one_dim,control_one_dim,y,roc_name)
	second_experiment_dynamic(sz_dynamic_one_dim,control_dynamic_one_dim,y)
