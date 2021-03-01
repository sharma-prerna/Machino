from django.shortcuts import render
from django.contrib import messages
#machine learning GUI project
from .algorithms import k_means_cluster as KMC
from .algorithms import knn as KNN 
from .algorithms import multi_linear_regression as Linear
from .algorithms import multi_logistic_regression as Logistic
from .algorithms import naive_bayes as NB
from .algorithms import deep_neural_network_model as DNN
from .algorithms import ml_utils as Utils
import numpy as np
import pandas as pd

# Create your views here.
def home(request):
	return render(request, 'machino/home.html')

#parameters
dataset = None
split_percent = None
status=X_train=Y_train=X_test=Y_test=last=None
test_data = train_data = columns =  None
#hyperparameters
k = None
layers = None
learning_rate = None
epochs = None
activation_function = None
all_files = Utils.get_media_list()

def knn(request):
	global all_files
	if request.method=="POST":
		try:
			dataset = request.FILES["dataset"]
			msg, msg_stat, all_files = Utils.save_file_media(dataset)
			
			if msg_stat=="success":
				messages.success(request,msg)
			else:
				messages.warning(request,msg)
		except:
			messages.warning(request, "Failed to Upload!")
			return render(request, 'machino/knn.html', {'datasets':all_files})
	
	elif request.method=="GET":
		filename = request.GET.get("filelist")
		try:
			split_percent = int(request.GET.get("split"))
		except:
			split_percent = None
		
		try:
			k = int(request.GET.get("k"))
		except:
			k = None
		
		if k and split_percent:
			status,X_train,Y_train,X_test,Y_test,last = Utils.load_external_csv_dataset(filename,split_percent,"Classification")
		else:
			status = "Upload File"
		
		if status!="VALID DATASET":
			messages.warning(request, status)
			return render(request, 'machino/knn.html', {'datasets':all_files})

		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test
		response = {'test_shape': test_data.shape, 'train_shape': train_data.shape}

		#Z_test is the prediction
		Z_test = KNN.predict_knn(X_train,Y_train,X_test,k)
		test_accuracy = KNN.accuracy(Z_test,Y_test)*100

		#adding last column again
		test_data["Prediction for "+last] = np.squeeze(Z_test)
		response['test_data']= Utils.filter_data(test_data)

		response['stat'], response['confusion'] = Utils.plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
		response['plot'] = Utils.plot(train_data,test_data,last,"K Nearest Neighbors")
		messages.success(request, "WOHOO! MODEL TRAINED")
		
		#putting the current dataset at first in the srquence list
		temp = all_files[0]
		all_files[all_files.index(filename)]=temp
		all_files[0] = filename

		response['accuracy'] = round(test_accuracy,2)
		response['datasets']= all_files
		response['used_dataset'] = filename
		response['k_value'] = k
		response['split_value'] = split_percent
		return render(request,'machino/knn.html', response)
	return render(request, 'machino/knn.html', {'datasets':all_files})

def deep(request):
	global all_files
	if request.method=="POST":
		try:
			dataset = request.FILES["dataset"]
			msg, msg_stat, all_files = Utils.save_file_media(dataset)
			
			if msg_stat=="success":
				messages.success(request,msg)
			else:
				messages.warning(request,msg)
		except:
			messages.warning(request, "Failed to Upload!")
			return render(request, 'machino/deep.html', {'datasets':all_files})
	
	elif request.method=="GET":
		filename = request.GET.get("filelist")
		try:
			split_percent = int(request.GET.get("split"))
		except:
			print("split error")
			split_percent = None

		try:
			layers = list(map(int,request.GET.get("layers").split()))
		except:
			messages.warning(request, "Invalid Input of Hidden Layers")
			layers = None

		try:
			input_activation = request.GET.get("inp-activate")
		except:
			input_activation = None

		try:
			output_activation = request.GET.get("out-activate")
		except:
			output_activation = None

		try:
			learning_rate = float(request.GET.get("lrate"))
		except:
			print("learning_rate error")
			learning_rate = None

		try:
			iterations = int(request.GET.get("iter"))
		except:
			print("iterations error")
			iterations = None

		if split_percent and layers is not None and input_activation and output_activation and learning_rate and iterations:
			status,X_train,Y_train,X_test,Y_test,last = Utils.load_external_csv_dataset(filename,split_percent,"Classification")
		else:
			status = "Upload File"
		
		if status!="VALID DATASET":
			messages.warning(request, status)
			return render(request, 'machino/deep.html', {'datasets':all_files})

		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test
		response = {'test_shape': test_data.shape, 'train_shape': train_data.shape}
		response['classes'] = list(test_data[last].unique())

		#adding input layer and output layer
		layers.insert(0, X_test.shape[1])
		layers.append(1)

		if len(response['classes'])>2:
			print(response['classes'])
			messages.warning(request, "More than two classes found in dataset!")
			return render(request, 'machino/deep.html', {'datasets':all_files})

		train_data[last], test_data[last] = pd.Categorical(train_data[last]), pd.Categorical(test_data[last])
		response['cat_map'] = dict(enumerate(test_data[last].cat.categories))
		train_data[last], test_data[last] = train_data[last].cat.codes, test_data[last].cat.codes
		
		Y_train, Y_test = np.array(train_data[last].copy()), np.array(test_data[last].copy())
		Y_train,Y_test = Y_train.reshape(1,train_data.shape[0]), Y_test.reshape(1,test_data.shape[0])

		X_train,X_test=DNN.normalize_data(X_train,X_test)
		
		parameters,costs= DNN.deep_neural_network(layers,learning_rate,iterations,X_train.T,Y_train,input_activation,output_activation)
		
		Z_train,train_accuracy,A_train = DNN.predict(X_train.T,Y_train,parameters,input_activation,output_activation)
		
		Z_test,test_accuracy,A_test = DNN.predict(X_test.T,Y_test,parameters,input_activation,output_activation)
		
		train_data["Prediction for "+last] = np.squeeze(Z_train)
		test_data["Prediction for "+last] = np.squeeze(Z_test)

		response['stat'], response['confusion'] = Utils.plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
		response['cost_graph']=Utils.plot_cost(costs)

		#current file is at first position
		temp = all_files[0]
		all_files[all_files.index(filename)]=temp
		all_files[0] = filename

		#all layers

		response['datasets'] = all_files
		response['training_accuracy'] = round(train_accuracy*100,2)
		response['testing_accuracy'] = round(test_accuracy*100,2)
		response['test_data'] = Utils.filter_data(test_data)
		response['layers'] = layers
		response['iterations'] = iterations
		response['learning_rate'] = learning_rate
		response['split_value'] = split_percent
		messages.success(request, "WOHOO! MODEL TRAINED")
		return render(request,'machino/deep.html',response)

	return render(request, 'machino/deep.html', {'datasets':all_files})

def lin(request):
	global all_files
	if request.method=="POST":
		try:
			dataset = request.FILES["dataset"]
			msg, msg_stat, all_files = Utils.save_file_media(dataset)
			
			if msg_stat=="success":
				messages.success(request,msg)
			else:
				messages.warning(request,msg)
		except:
			messages.warning(request, "Failed to Upload!")
			return render(request, 'machino/lin.html', {'datasets':all_files})
	
	elif request.method=="GET":
		filename = request.GET.get("filelist")
		try:
			split_percent = int(request.GET.get("split"))
		except:
			print("split error")
			split_percent = None

		try:
			learning_rate = float(request.GET.get("lrate"))
		except:
			print("learning_rate error")
			learning_rate = None

		try:
			iterations = int(request.GET.get("iter"))
		except:
			print("iterations error")
			iterations = None

		if split_percent and learning_rate and iterations:
			status,X_train,Y_train,X_test,Y_test,last = Utils.load_external_csv_dataset(filename,split_percent,"Regression")
		else:
			status = "Upload File"
		
		if status!="VALID DATASET":
			messages.warning(request, status)
			return render(request, 'machino/lin.html', {'datasets':all_files})

		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test
		response = {'test_shape': test_data.shape, 'train_shape': train_data.shape}
		
		Y_train, Y_test = np.array(Y_train.copy()),np.array(Y_test.copy())
		Y_train = Y_train.reshape(1,train_data.shape[0])
		Y_test = Y_test.reshape(1,test_data.shape[0])

		X_train = Linear.normalize_data(X_train)
		X_test = Linear.normalize_data(X_test)

		w,b,costs = Linear.linear_regression(X_train.T,Y_train,learning_rate,iterations)

		Z_train = np.dot(w,X_train.T)+b
		response['train_mse'] = round(np.sum((Z_train-Y_train)**2)/Y_train.shape[1],2)

		Z_test = np.dot(w,X_test.T)+b
		response['test_mse'] = round(np.sum((Z_test-Y_test)**2)/Y_test.shape[1],2)

		train_data["Prediction for "+ last]=np.round(np.squeeze(Z_train),2)
		test_data["Prediction for "+last]=np.round(np.squeeze(Z_test),2)

		response['plot'] = Utils.plot(train_data,test_data,last,"Linear Regression")
		response['cost_graph'] = Utils.plot_cost(costs)

		#current file is at first position
		temp = all_files[0]
		all_files[all_files.index(filename)]=temp
		all_files[0] = filename

		response['datasets'] = all_files
		response['test_data'] = Utils.filter_data(test_data)
		response['iterations'] = iterations
		response['learning_rate'] = learning_rate
		response['split_value'] = split_percent
		messages.success(request, "WOHOO! MODEL TRAINED")
		return render(request,'machino/lin.html',response)
	return render(request, 'machino/lin.html',{'datasets':all_files})
	render(request, 'machino/lin.html',{'datasets':all_files})

def log(request):
	global all_files
	if request.method=="POST":
		try:
			dataset = request.FILES["dataset"]
			msg, msg_stat, all_files = Utils.save_file_media(dataset)
			
			if msg_stat=="success":
				messages.success(request,msg)
			else:
				messages.warning(request,msg)
		except:
			messages.warning(request, "Failed to Upload!")
			return render(request, 'machino/log.html', {'datasets':all_files})
	
	elif request.method=="GET":
		filename = request.GET.get("filelist")
		try:
			split_percent = int(request.GET.get("split"))
		except:
			print("split error")
			split_percent = None

		try:
			learning_rate = float(request.GET.get("lrate"))
		except:
			print("learning_rate error")
			learning_rate = None

		try:
			iterations = int(request.GET.get("iter"))
		except:
			print("iterations error")
			iterations = None

		if split_percent and learning_rate and iterations:
			if filename.endswith('Gate.csv'):
				split_percent = -1
			status,X_train,Y_train,X_test,Y_test,last = Utils.load_external_csv_dataset(filename,split_percent,"Classification")
		else:
			status = "Upload File"
		
		if status!="VALID DATASET":
			messages.warning(request, status)
			return render(request, 'machino/log.html', {'datasets':all_files})

		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test
		response = {'test_shape': test_data.shape, 'train_shape': train_data.shape}
		response['classes'] = list(test_data[last].unique())

		if len(response['classes'])>2:
			print(response['classes'])
			messages.warning(request, "More than two classes found in dataset!")
			return render(request, 'machino/log.html', {'datasets':all_files})

		train_data[last], test_data[last] = pd.Categorical(train_data[last]), pd.Categorical(test_data[last])
		response['cat_map'] = dict(enumerate(test_data[last].cat.categories))
		train_data[last], test_data[last] = train_data[last].cat.codes, test_data[last].cat.codes
		
		Y_train, Y_test = np.array(train_data[last].copy()), np.array(test_data[last].copy())
		Y_train,Y_test = Y_train.reshape(1,train_data.shape[0]), Y_test.reshape(1,test_data.shape[0])

		X_train = Logistic.normalize_data(X_train)
		X_test = Logistic.normalize_data(X_test)
		
		w,b,costs = Logistic.logistic_regression(X_train.T,Y_train,learning_rate,iterations)

		Z_train,train_accuracy,A_train = Logistic.predict(X_train.T,Y_train,w,b)
		Z_test,test_accuracy,A_test = Logistic.predict(X_test.T,Y_test,w,b)
		
		train_data["Prediction for "+last] = np.squeeze(Z_train)
		test_data["Prediction for "+last] = np.squeeze(Z_test)

		response['stat'], response['confusion'] = Utils.plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
		response['cost_graph']=Utils.plot_cost(costs)
		response['decision'] = Utils.plot_decision_boundary(test_data,w,b,last)
		#current file is at first position
		temp = all_files[0]
		all_files[all_files.index(filename)]=temp
		all_files[0] = filename

		response['datasets'] = all_files
		response['training_accuracy'] = round(train_accuracy*100,2)
		response['testing_accuracy'] = round(test_accuracy*100,2)
		response['test_data'] = Utils.filter_data(test_data)
		response['iterations'] = iterations
		response['learning_rate'] = learning_rate
		response['split_value'] = split_percent
		messages.success(request, "WOHOO! MODEL TRAINED")
		return render(request,'machino/log.html',response)
	return render(request, 'machino/log.html',{'datasets':all_files})

def naive(request):
	global all_files
	if request.method=="POST":
		try:
			dataset = request.FILES["dataset"]
			msg, msg_stat, all_files = Utils.save_file_media(dataset)
			
			if msg_stat=="success":
				messages.success(request,msg)
			else:
				messages.warning(request,msg)
		except:
			messages.warning(request, "Failed to Upload!")
			return render(request, 'machino/naive.html', {'datasets':all_files})
	
	elif request.method=="GET":
		filename = request.GET.get("filelist")
		try:
			split_percent = int(request.GET.get("split"))
		except:
			print("split error")
			split_percent = None

		if split_percent:
			status,X_train,Y_train,X_test,Y_test,last = Utils.load_external_csv_dataset(filename,split_percent,"Classification")
		else:
			status = "Upload File"
		
		if status!="VALID DATASET":
			messages.warning(request, status)
			return render(request, 'machino/naive.html', {'datasets':all_files})

		train_data, test_data = X_train.copy(), X_test.copy()
		train_data[last], test_data[last] = Y_train, Y_test
		response = {'test_shape': test_data.shape, 'train_shape': train_data.shape}
		classes = list(test_data[last].unique())
		
		marginal_prob,mean_std= NB.gaussian_naive_bayes_classiefier(train_data,classes,last)
			
		Z_train,train_accuracy = NB.predict(X_train,Y_train,mean_std,marginal_prob,classes)
	
		Z_test,test_accuracy = NB.predict(X_test,Y_test,mean_std,marginal_prob,classes)

		test_data["Prediction for "+last]=np.squeeze(Z_test)
		train_data["Prediction for "+last]=np.squeeze(Z_train)

		response['stat'], response['confusion'] = Utils.plot_confusion_matrix(test_data[last],test_data["Prediction for "+last])
		
		temp = all_files[0]
		all_files[all_files.index(filename)]=temp
		all_files[0] = filename

		response['datasets'] = all_files
		response['training_accuracy'] = round(train_accuracy*100,2)
		response['testing_accuracy'] = round(test_accuracy*100,2)
		response['test_data'] = Utils.filter_data(test_data)
		response['split_value'] = split_percent
		response['classes'] = classes
		messages.success(request, "WOHOO! MODEL TRAINED")
		return render(request,'machino/naive.html',response)
	return render(request, 'machino/naive.html',{'datasets':all_files})

def kmc(request):
	global all_files
	if request.method=="POST":
		try:
			dataset = request.FILES["dataset"]
			msg, msg_stat, all_files = Utils.save_file_media(dataset)
			
			if msg_stat=="success":
				messages.success(request,msg)
			else:
				messages.warning(request,msg)
		except:
			messages.warning(request, "Failed to Upload!")
			return render(request, 'machino/kmc.html', {'datasets':all_files})
	
	elif request.method=="GET":
		filename = request.GET.get("filelist")
		try:
			k = int(request.GET.get("k"))
		except:
			k = None


		try:
			cluster_names = request.GET.get("clust").strip().split()
			print(cluster_names)
		except:
			cluster_names = None
		
		if len(cluster_names)!=k and k is not None:
			messages.warning(request, "Total count of Cluster-Seeds is not equal to Input K")
			return render(request, 'machino/kmc.html', {'datasets':all_files,'k_value':k})
		
		if len(set(cluster_names))<len(cluster_names):
			messages.warning(request, "Cluster Seeds are not unique")
			cluster_names = None
		try:
			if not all([int(clust)>0 for clust in cluster_names]):
				messages.warning(request, "Cluster Seeds are not positive integers")
				cluster_names = None
		except:
			messages.warning(request, "Cluster Seeds are not positive integers")
			cluster_names = None
		
		if k is not None and cluster_names is not None:
			status,X_train,Y_train,_,_,last = Utils.load_external_csv_dataset(filename,0,"Clustering")
		else:
			status = "Upload File"

		if status!="VALID DATASET":
			messages.warning(request, status)
			return render(request, 'machino/kmc.html', {'datasets':all_files})

		if len(cluster_names)>(X_train.shape[0]):
			messages.warning(request, "Cluster Seeds exceeds the total number of samples")
			return render(request, 'machino/kmc.html', {'datasets':all_files, 'train_shape':X_train.shape})

		label ="Seeds"
		X_train[label] = list(map(str,range(1,X_train.shape[0]+1)))
		response={'train_shape':X_train.shape}
		
		clustered_data = KMC.k_means_clustering(X_train,cluster_names,label)
		response['clusters'] = list(clustered_data["cluster"].unique())
		response['plot'] = Utils.plot(clustered_data,None,"cluster","K Means Clustering")

		temp = all_files[0]
		all_files[all_files.index(filename)]=temp
		all_files[0] = filename

		response['datasets'] = all_files
		response['test_data'] = Utils.filter_data(clustered_data)
		response['k_value'] = k
		messages.success(request, "WOHOO! MODEL TRAINED")
		
		return render(request,'machino/kmc.html',response)
	return render(request, 'machino/kmc.html',{'datasets':all_files})