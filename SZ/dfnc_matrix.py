import numpy as np
import matplotlib.pyplot as plt

def window_per_people(filemane):
	filename1 = filename
	# Load file
	brain_time_data = np.loadtxt(open(filename1, "rb"), delimiter=",") #Each row is a region, each column is a time point
	
	line_col=np.shape(brain_time_data)
	line=line_col[0] #number of line
	col=line_col[1] #number of column

	shape =(96,50) #line and column 
	window_one= np.zeros(shape)
	window_two= np.zeros(shape)
	window_three= np.zeros(shape)
		
	for i in range(0,line):
		line_btd=brain_time_data[i] # we take the line 
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

def plot_window(obj_patient,state_name):
	wind_with_pat= obj_patient
	state=str(state_name)		
	fig=plt.figure()
	length= len(obj_patient)
	for i in range(0,length):
		l=wind_with_pat[i]
		set_window=l[0]
		patient_name=l[1]
		for window in range(len(set_window)):
			# Compute functional connectivity
			corr = np.corrcoef(set_window[window]) #This is the static functional 		connectivity, to get the dynamical you have to select only few columns
			mask =  np.tri(corr.shape[0], k=-1)
			corr = np.ma.array(corr, mask=mask) # mask out the lower triangle
			fig.add_subplot(length,3,window+1)
			plt.imshow(corr, cmap='jet',origin ='upper',extent=[0., 96., 0., 96.],interpolation='nearest')
			plt.title("Window "+str(window+1))
		if state =='schi':
			plt.suptitle('Patient Schizophrenia number '+str(patient_name)) 
		
		elif state =='control':
			plt.suptitle('Patient Control healthy number '+str(patient_name))

if __name__=="__main__":
	#data schizophrenia
	filename_sz=["../SZ/0040001.csv","../SZ/0040007.csv","../SZ/0040015.csv","../SZ/0040037.csv","../SZ/0040080.csv","../SZ/0040089.csv","../SZ/0040099.csv","../SZ/0040108.csv","../SZ/0040034.csv","../SZ/0040105.csv","../SZ/0040122.csv","../SZ/0040096.csv","../SZ/0040132.csv","../SZ/0040100.csv","../SZ/0040142.csv","../SZ/0040145.csv"]
	#data control healthy
	filename_control=["0040027.csv","0040026.csv","0040043.csv","0040024.csv","0040038.csv","0040035.csv","0040033.csv","0040062.csv","0040061.csv","0040058.csv","0040056.csv","0040053.csv","0040051.csv","0040030.csv","0040063.csv","0040065.csv","0040066.csv","0040067.csv","0040068.csv"]
	#for filename in filename_control: # for healthy control
	for filename in filename_sz: #for schizophrenia
		obj_patient=[]
		win_set= window_per_people(filename)
		obj_patient.append((win_set,filename))
		state_name= 'schi'
		#state_name= 'control'
		plot_window(obj_patient,state_name)	
	plt.colorbar()
	plt.show()
	
