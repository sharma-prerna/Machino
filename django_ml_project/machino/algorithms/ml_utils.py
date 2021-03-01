#LOAD DATASETS OF DIFFERENT TYPES

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_float_dtype

from django.core.files.storage import FileSystemStorage
from pathlib import Path
import io
import os
import urllib, base64
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(BASE_DIR)
media_root = os.path.join(BASE_DIR, 'media')

def load_external_csv_dataset(filename,split_percent,algo_type):
	try:
		filepath = os.path.join(media_root, filename)
		df = pd.read_csv(filepath)
		cols = list(df.columns)
		last = cols[-1]
	except Exception as e:
		return f'{e}',None,None,None,None,None

	if df.isnull().values.any():
		return "NAN VALUES IN DATASET",None,None,None,None,None

	#check whether all input data is numeric or not
	for column in cols[:-1]:
		if not is_numeric_dtype(df[column]):
			return "NON NUMERIC VALUE FOUND IN DATASET (EXCLUDING OUTPUT COLUMN)",None,None,None,None,None
	if is_float_dtype(df[last]) and algo_type=="Classification": 
		return "OUTPUT COLUMN DOES NOT CONTAIN CATEGORICAL VALUES",None,None,None,None,None
	if not is_float_dtype(df[last]) and algo_type=="Regression": 
		return "OUTPUT COLUMN DOES NOT CONTAIN CONTINUOUS NUMERIC VALUES",None,None,None,None,None
		
	status="VALID DATASET"
	if split_percent<0:
		train_output = df[last]
		train_input = df.drop([last],axis=1)
		test_input = train_input.copy()
		test_output = train_output.copy()
		return status,train_input,train_output,test_input,test_output,last
	#randomly shuffle the data first
	np.random.seed(2)
	df = df.sample(frac=1).reset_index(drop=True)
	
	#splitting the training data and testing data
	test_data_size = (df.shape[0]*split_percent)//100
	r_no=np.random.randint(df.shape[0]-test_data_size)
	
	test_data= df[r_no:r_no+test_data_size]
	train_data=df.drop(range(r_no,r_no+test_data_size))
		
	#splitting ouptut and input
	train_output=train_data[last]
	test_output=test_data[last]

	train_input=train_data.drop([last],axis=1)
	test_input=test_data.drop([last],axis=1)

	return status,train_input,train_output,test_input,test_output,last

def save_file_media(file):
	all_files = get_media_list()
	if file.name in all_files:
		return "File Already Exist","warning",all_files
	
	fs = FileSystemStorage()
	fs.save(file.name, file)
	all_files  = get_media_list()
	return "File Uploaded Successfully","success",all_files

def get_media_list():
	return os.listdir(media_root)

def plot(train_data,test_data,last,type):
	
	if type=="Linear Regression":
		fig,ax = plt.subplots(figsize=(7,4))
		cols = list(train_data.columns)
		x1 = np.array(train_data[cols[0]])
		y1 = np.array(train_data[cols[-2]])
		x2 = np.array(test_data[cols[0]])
		y2 = np.array(test_data[cols[-2]])

		#computing regression line
		z = np.squeeze(test_data[cols[-1]])
		x_min, z_min = x2.min(),z.min()
		x_max, z_max = x2.max(),z.max()
		
		x_line = np.linspace(x_min,x_max,num=len(np.squeeze(x2)))
		z_line = np.linspace(start=z_min,stop=z_max,num=len(np.squeeze(z)))
		
		ax.scatter(x1,y1,color='blue',label='Training data points',edgecolor='white')
		ax.scatter(x2,y2,color='green',label='Testing data points',edgecolor='white')

		ax.plot(x_line,z_line,color='g',label='Regression Line')
		ax.set_title('Linear Regression')
		ax.set_xlabel(cols[0])
		ax.set_ylabel(cols[-2])
		ax.legend(loc='upper right')
		ax.set_facecolor("aliceblue")
		img = create_img_of_plot(fig)
		plt.close()
		return img
		
	elif type=="K Nearest Neighbors":
		cols = list(train_data.columns)
		fig,ax = plt.subplots(figsize=(7,4))
		sns.scatterplot(x=cols[0],y=cols[1],data=train_data,hue=last)
		sns.scatterplot(x=cols[0],y=cols[1],s=150,data=test_data,color='black',marker='*',legend=False,label='Test Instances')
		ax.set_xlabel(cols[0])
		ax.set_ylabel(cols[1])
		ax.set_facecolor("aliceblue")
		ax.legend(loc='upper right')

		#create a img of the plot and return it back
		img = create_img_of_plot(fig)
		plt.close()
		return img


	elif type=="K Means Clustering":
		cols = list(train_data.columns)
		fig,ax = plt.subplots()
		sns.scatterplot(x=cols[0],y=cols[1],hue=last,data=train_data)	
		ax.set_title('K Means Clustering')
		ax.set_xlabel(cols[0])
		ax.set_ylabel(cols[1])
		ax.set_facecolor("lavender")
		ax.legend()
		img = create_img_of_plot(fig)
		plt.close()
		return img


def plot_confusion_matrix(actual,predicted):
	#print(actual, predicted)
	classes = list(actual.unique())
	confusion_matrix = pd.crosstab(actual,predicted,rownames=['Actual'],colnames=['Predicted'],margins=True)
	#print(confusion_matrix)
	fig, ax = plt.subplots(figsize=(7,4))
	sns.heatmap(confusion_matrix,annot=True,ax=ax)
	img = create_img_of_plot(fig)
	plt.close()

	if len(classes)<=2:
		try:
			TP = confusion_matrix[classes[0]][classes[0]]
		except:
			TP = 0

		try:
			TN = confusion_matrix[classes[1]][classes[1]]
		except:
			TN = 0

		try:
			FP = confusion_matrix[classes[1]][classes[0]]
		except:
			FP = 0

		try:
			FN = confusion_matrix[classes[0]][classes[1]]
		except:
			FN = 0

		recall = round(TP/(TP+FN),4) if TP or FN else np.nan
		sensitivity = round(TP/(TP+FN),4) if TP or FN else np.nan
		specificity = round(TN/(FP+TN),4) if TN or FP else np.nan
		precision = round(TP/(TP+FP),4) if TP or FP else np.nan
		Fscore = round(2/((1/recall) + (1/precision)),4) if recall and precision else np.nan

		stat_data = zip(["Sensitivity","Specificity","Precision","Recall","F1 Score"],[sensitivity,specificity,precision,recall,Fscore])
	else:
		stat_data=[]
	return stat_data, img

def plot_cost(costs):
	fig, ax = plt.subplots(figsize=(7,4))
	ax.plot(costs)
	ax.set_xlabel("Iterations")
	ax.set_ylabel("Cost")
	ax.set_facecolor("aliceblue")
	ax.grid()
	ax.set_title("Cost Reduction Graph")
	img = create_img_of_plot(fig)
	plt.close()
	return img

def plot_decision_boundary(data,W,b,last):
	cols = list(data.columns)
	#normlize the data
	from . import multi_logistic_regression as Logistic
	data[cols[0]], data[cols[1]] = Logistic.normalize_data(data[cols[0]]), Logistic.normalize_data(data[cols[1]])
	X, Y = data[cols[0]], data[cols[1]]
	x1 = np.arange(X.min()-0.1, X.max()+0.1, 0.1)
	x2 = np.arange(Y.min()-0.1, Y.max()+0.1, 0.1)
	
	w1, w2 = W[:,0],W[:,1]
	x, y = np.meshgrid(x1,x2)
	f = lambda x, y: Logistic.sigmoid(x*w1+ y*w2 + b)
	z= f(x,y)

	fig , ax = plt.subplots(figsize=(7,4))
	sns.scatterplot(x=cols[0],y=cols[1],data=data,s=70,hue=last,palette='dark')
	ax.contourf(x,y,z,alpha=0.6,levels=0,cmap='inferno')
	ax.set_xlabel(cols[0])
	ax.set_ylabel(cols[1])
	ax.legend()
	ax.set_title("Decision Boundary")
	img = create_img_of_plot(fig)
	plt.close()
	return img
	
def create_img_of_plot(fig):
	buff = io.BytesIO()
	fig.savefig(buff, format='png')
	buff.seek(0)
	string  = base64.b64encode(buff.read())
	img = urllib.parse.quote(string)
	return img

def filter_data(data):
	columns = list(data.columns)
	arr = np.array(data)
	lst = list()
	lst.append(columns)

	for i in range(data.shape[0]):
		tmp = list()
		for j in range(data.shape[1]):
			if arr[i][j]!="\n":
				tmp.append(arr[i][j]) 
		lst.append(tmp)
	return lst