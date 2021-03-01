''' K means clustering : Unsupervised learning :'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(filename):
	df = pd.read_csv(filename)
	#print("\n------------ UNCLUSTERED DATA ---------\n")
	#print(df.head())
	#print("---------------------------------------\n")
	cols = list(df.columns)
	label = cols[0]
	valid_cluster_names = np.array(df[label].unique())

	return df,label,valid_cluster_names

def euclid_distance(X,Y):
	# Transpose the data so that we can compute distance along features
	##print(Y)
	##print(X)
	Y = Y.reshape(1,X.shape[1])
	dist= np.sqrt(np.sum(np.power(X-Y,2),axis=1,keepdims=True))
	return dist


def manhattan_distance(X,Y):
	Y = Y.reshape(1,X.shape[1])
	dist=np.sum(np.abs(X-Y),axis=1,keepdims = True)
	return dist



def update_clusters(data,cluster_names):

	new_cluster_seeds={}
	#print("---------------------------------------\n")
	for i,cluster in enumerate(cluster_names):
		samples=data[(data["cluster"]==cluster)]
		samples=samples.drop(["cluster"],axis=1)
		new_cluster_seeds["cluster"+str(i+1)]=np.array(samples.mean())

	assert(new_cluster_seeds!=None)

	return new_cluster_seeds

def assign_initial_clusters(X_data,cluster_names,label):
	cluster_seeds= {}
	for i,clust in enumerate(cluster_names):

		assert(X_data[X_data[label]==clust].shape[0]==1)
		sample = X_data[X_data[label]==clust]
		sample = sample.drop([label],axis=1)
		cluster_seeds["cluster"+str(i+1)]= np.array(sample)

	distances = np.empty((len(cluster_names),X_data.shape[0]))
	Y_data = []

	#calculating euclidean distance of data points from all the clusters
	#first remove all non numerical values from the data
	num_data = np.array(X_data.drop([label],axis=1))

	for i,clust in enumerate(cluster_seeds.values()):
		dist = euclid_distance(num_data,np.array(clust))
		distances[i]= np.squeeze(dist)

	for i in range(len(cluster_names)):
		cluster_names[i] = "C"+str(i+1)

	distances = np.argmin(distances,axis=0)
	for i,index in enumerate(distances):
		Y_data.append(cluster_names[index])

	X_data["cluster"] = Y_data

	#print(X_data) 
	#print()
	assert(cluster_seeds!=None)

	return X_data,cluster_seeds

def k_means_clustering(X_data,cluster_names,label):

	X_data,cluster_seeds = assign_initial_clusters(X_data,cluster_names,label)
	##print("cluster_names: {}".format(cluster_names))
	change = True
	#numerical data for mathematical computation
	X_data = X_data.drop([label],axis = 1)
	num_data = np.array(X_data.drop(["cluster"],axis=1)) 
	##print(num_data)
	while change:
		change = False
		old_clusters = X_data["cluster"]
		cluster_seeds = update_clusters(X_data,cluster_names)

		new_clusters = []
		distances = np.empty((len(cluster_names),X_data.shape[0]))

		#print(cluster_seeds)
		for i,clust in enumerate(cluster_seeds.values()):
			#print("cluster{} : {}".format(i,clust))
			dist = euclid_distance(num_data,clust)
			distances[i]= np.squeeze(dist)

		distances = np.argmin(distances,axis=0)

		for i,index in enumerate(distances):
			new_clusters.append(cluster_names[index])

		if all(old_clusters==new_clusters) == False:
			change = True
			X_data["cluster"] = new_clusters
			

		#print()
		#print(X_data)

	return X_data

if __name__ == "__main__":

	#loading unclustered data
	data,label,valid_cluster_names = load_dataset("student_marks.csv")

	#print()
	#print("Enter the number of clusters :",end=" ")
	try:
		k = int(input().strip())
		#print()

	except:
		#print("Please enter a positive integer number")
		exit()

	if k<=0:
		#print("Please enter a positive integer number")
		exit()
	elif k > len(valid_cluster_names):
		#print("Please enter value of k between: 1 to {}".format(len(valid_cluster_names)))
		exit()

	#print("Valid choices for clusters : {} ".format(valid_cluster_names))
	#print("Enter valid {} cluster seed names:".format(k),end=" ")
	cluster_names = input().strip().split()

	if k!=len(set(cluster_names)) or k > len(valid_cluster_names):
		#print("Please enter exactly {} distinct seeds".format(k))
		exit()

	#check for wrong input for cluster names
	for clust in cluster_names:
		if clust not in valid_cluster_names:
			#print("Oops! Invalid cluster name :(")
			#print("Valid cluster names are given above")
			exit()

	clustered_data = k_means_clustering(data,cluster_names,label)
	clusters = clustered_data["cluster"]
	#plot the clusters
	
	#print("\n----------------------Data is clustered successfully-----------------------")






