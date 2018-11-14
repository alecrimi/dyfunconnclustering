# Code written by R.Muntari and A.Andriamananjara to compute 5-window dynamic functional connectivity in a given dataset

import numpy as np
import os
import math as mt
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import auc

def window_per_people(subject):

	line_col=np.shape(subject)
	line=line_col[0] #number of line
	col=line_col[1] #number of column 

        size_w = 30
	shape =(96,size_w) #number of line and column 
	window_one= np.zeros(shape)
	window_two= np.zeros(shape)
	window_three= np.zeros(shape)
	window_four= np.zeros(shape) 
        window_five= np.zeros(shape) 
        
	for i in range(0,line):
		line_btd=subject[i] # we take each line 
		for j in range(0,col):
			if (j<size_w and j>=0):
				w=line_btd[j]
				window_one[i][j]=w # put in the first window
                        
			elif (j<size_w*2 and j>=size_w): 
				w=line_btd[j]
				window_two[i][j-size_w*2]=w # put in the second window
                       
			elif (j<size_w*3 and j>=size_w*2): #
				w=line_btd[j]
				window_three[i][j-size_w*3]=w # put in the third window  
                        
                        elif (j<size_w*4 and j>=size_w*3):
				w=line_btd[j]
				window_four[i][j-size_w*4]=w # put in the third window
                        
                        elif (j<size_w*5 and j>=size_w*4):
				w=line_btd[j]
				window_five[i][j-size_w*5]=w # put in the third window
                    	set_window=[window_one, window_two, window_three  , window_four, window_five ] 
	return set_window

	
# this function compute the correlation matrix

def static_connectivity_matrix(subject):
	corr= np.corrcoef(subject)	
	return corr 
	
#this function flat the connectivity matrix in 1D
def features(correlation_matrix_static):

	corr= correlation_matrix_static
	vec_1D_per_subject_static=[0]
		
	#we take triangular superior of the correlation matrix 
	triangular_sup = np.triu(corr,k=1)
	'''
	#transformation of the triangular matrix in 1D vector  
	vec_1D_per_subject_static=vec_1D_per_subject_static+list(triangular_sup[np.triu_indices(96)])
	
	del vec_1D_per_subject_static[0]# delete the initial value
        '''	 
        rows, cols = np.nonzero(triangular_sup)
        vec_1D_per_subject_static = triangular_sup[rows, cols]
 
	#the function nan_to_num replace nan with zero and inf with large finite numbers. 
	vec_1D_per_subject_static=np.nan_to_num(vec_1D_per_subject_static)
	
	return vec_1D_per_subject_static
	
def dynamical_connectivity(win_set,folder,i):
	set_window= win_set
	
	dynamic_mat_list=[]
	counter = 0
	for window in range(len(set_window)):
		
		# we compute the static connectivity per window which makes the dynamic		
		corr = static_connectivity_matrix(set_window[window]) 
		np.savetxt(folder+"_"+str(i)+"_"+str(counter)+".csv", corr, delimiter=",")
		# the function features flats the matrix triangular superior in dynamical connect...
		a=list(features(corr))
		dynamic_mat_list=dynamic_mat_list+a
		counter = counter +1
	return dynamic_mat_list
	
# Cross validation
def Leave_one_out(X,y):

	# conversion of lists to array
	#X=np.asarray(X)
	#y=np.asarray(y)
	
	loo = LeaveOneOut()
	n = np.shape(y)[0] -1
 
 	X_train=[[]*n for x in xrange(n)]
	X_test=[]
	y_train=[]
	y_test=[]
	count=0
	score= np.zeros(n+1)
	
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
		
		#print('X train length',len(X_train),' y train length',len(y_train))
		clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
		#print(X_test.shape)
		#print(len(X_test))
		probs = clf.predict_proba(X_test.reshape(1,-1))
		score[count] = probs[:,0]
		count+=1
		
		#clear the lists
	
		X_train=[[]*n for x in xrange(n)]
		X_test=[]
		y_train=[]
		y_test=[]
	
	return score
'''
	loo = LeaveOneOut()
	score = np.zeros(144)
	count = 0
	train_X=np.zeros(143)
	train_y=np.zeros(143)
	test_X=[0]
	test_y=[0]
	for train_index, test_index in loo.split(X):
		#print(train_index.shape,test_index.shape)
		for i in range(len(train_index)):
			train_X[i]=X[train_index[i]]
			train_y[i]=y[train_index[i]]
		test_X=X[test_index[0]]
		test_y=y[test_index[0]]
		clf = svm.SVC(kernel='linear', probability=True).fit(train_X, train_y)
		probs = clf.predict_proba(test_X)
		score[count] = probs[:,0]
		count+=1

	return score 
'''	

'''
	
	print(' lenght xtrain',len(X_train),' lenght xtest',len(X_test),' lenght ytrain', len(y_train),' lenght ytest', len(y_test))
	
	return X_train, X_test, y_train , y_test
'''
	
def Alessandro_ROC(y_preditc_proba,ytest):
	score=y_preditc_proba
	roc_x = []
	roc_y = []
	min_score = min(score)
	max_score = max(score)
	thr = np.linspace(min_score, max_score, 30)
	FP=0
	TP=0
	P = sum(ytest)
	N = len(ytest) - P
	for (i,T) in enumerate(thr):
		for i in range(0, len(score)):
			if (score[i] > T):

				if (ytest[i]==1):
					TP = TP + 1
				if (ytest[i]==0):
					FP = FP + 1
		
		roc_x.append(FP/float(N))
		roc_y.append(TP/float(P))
		FP=0
		TP=0

	return  roc_x,roc_y

#Plotting of a ROC curve for a specific class
def plot_ROC(roc_x,roc_y):

	#roc_x.insert(0,0)
	#roc_y.insert(0,0)
 	#roc_x =[0]+roc_x+[1] #np.append(roc_x, [1])
	#roc_y =[0]+roc_y+[1] #np.append(roc_y, [1])
	
        list_x= roc_x  
	list_y=  roc_y  
	#list_x= np.insert(roc_x, 1, 0) 
	#list_y= np.insert(roc_y, 1, 0) 
	#list_x= np.insert(roc_x, 0, len(list_x)) 
	#list_y= np.insert(roc_y, 0, len(list_y)) 
  
	roc_auc= auc(list_x, list_y)
	lw = 2
	plt.plot(list_x,list_y, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

#This function does the experiment with the static connectivity
def run_experiment(sz_one_dim, control_one_dim, y):
	
	#matrix of all subject in schizophrenia and control healthy
	allSubjects_dFC_matrices_SZ=sz_one_dim
	allSubjects_dFC_matrices_control=control_one_dim
	
	# concatenation of the two matrix i.e the whole datset
	#X=sum([allSubjects_dFC_matrices_SZ+allSubjects_dFC_matrices_control],[])
	X =  np.concatenate((allSubjects_dFC_matrices_SZ, allSubjects_dFC_matrices_control), axis=0)
        print(np.shape(X))
	y_preditc_proba=Leave_one_out(X,y)
	
	#y_preditc_proba=clf.predict_proba(xtrain)[:,1]
	
	roc_x,roc_y =Alessandro_ROC(y_preditc_proba,y)
	#roc_x,roc_y =Alessandro_ROC(y_preditc_proba,ytrain)
	#fpr, tpr, _ = roc_curve(ytest, y_preditc_proba)
	#fpr, tpr, _ = roc_curve(ytrain, y_preditc_proba)
	
	#we plot the ROC curve
	plot_ROC(roc_y,roc_x)
	#plot_ROC(tpr,fpr)

'''	
	X_train, X_test, y_train , y_test=Leave_one_out(X,y)
	
	#Classification SVM with kernel=linear
	clf = svm.SVC(kernel='linear', probability=True)
	clf.fit(X_train, y_train) 
	
	#we transform the data xtest in probability
	y_preditc_proba=clf.predict_proba(X_test)[:,1] #[:,1] means the probability of label being 1
'''
 

if __name__=="__main__":
	sz_one_dim=[]			
	sz_dynamic_one_dim=[]		
	
	control_one_dim=[]
	control_dynamic_one_dim= []
	
	#data schizophrenia
        counter = 0
	filename_sz = os.listdir("ASD")
	for filename in filename_sz: 
		subject = np.loadtxt(open("ASD/"+filename, "rb"), delimiter=",")
                '''
		# create dataset static FC
                static_CM = static_connectivity_matrix(subject)
		sz_one_dim.append(features(static_CM))
		np.savetxt("SZ_static"+str(counter)+".csv", static_CM, delimiter=",")
                '''
                # create dataset dynamic FC
		win_set = window_per_people(subject)
		sz_dynamic_one_dim.append(dynamical_connectivity(win_set,"ASD",counter))
	        counter = counter +1
		
        #data control healthy
	filename_control = os.listdir("control")
        counter = 0
	for filename in filename_control:
		subject = np.loadtxt(open("control/"+filename, "rb"), delimiter=",")
                '''
		# create dataset static FC
	 	static_CM = static_connectivity_matrix(subject)
                np.savetxt("control_static"+str(counter)+".csv", static_CM, delimiter=",")
		control_one_dim.append(features(static_CM))
                '''
                # create dataset dynamic FC		
		win_set = window_per_people(subject)
		control_dynamic_one_dim.append(dynamical_connectivity(win_set,"control",counter))
                counter = counter +1 
	#print np.shape(control_dynamic_one_dim)

	#Labels ASD
	y=np.zeros(54)
	y[0:30]=1    #subject get the schizophrenia
	y[31:54]=0  #subject healthy

	#Labels SZ
#	y=np.zeros(144)
#	y[0:70]=1    #subject get the schizophrenia
#	y[71:144]=0  #subject healthy
 
	#the two experiments
        #Static FC
	#run_experiment(sz_one_dim,control_one_dim,y)
        #Dynamic FC
	run_experiment(sz_dynamic_one_dim,control_dynamic_one_dim,y)
