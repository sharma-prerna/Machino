
def k_means_clustering_model(algo_type):
	ds = st.sidebar.selectbox("Type of Dataset",["Iris Flower Dataset","Wine Quality Dataset","Other"])
	dataset = False
	status =None
	
	if ds=="Iris Flower Dataset":
		dataset = True
		status,X_train,Y_train,_,_,last = LOAD.load_internal_csv_dataset("Iris.csv",0,"Clustering")
	
	elif ds=="Wine Quality Dataset":
		dataset = True
		status,X_train,Y_train,_,_,last = LOAD.load_internal_csv_dataset("wine.csv",0,"Clustering")
	
	elif ds=="Other":
		st.sidebar.markdown("""<ul><li>File Format: CSV</li><li>Must be preprocessed</li> <li>Must not contain any NAN values</li>
			<li>Must not contain invalid values (Combination of numeric and non-numeric)</li>
			<li>Assumption: First n-1 columns are numeric and last column (output) can be either numeric or non-numeric </li>""",unsafe_allow_html=True)

		dataset = st.sidebar.file_uploader("Upload Dataset")
		status,X_train,Y_train,_,_,last = LOAD.load_external_csv_dataset(dataset,0,"Clustering")
	
	if dataset and status=="VALID DATASET":
		st.sidebar.success(status)
		label ="Seeds"
		X_train[label] = list(map(str,range(1,X_train.shape[0]+1)))
		
		st.sidebar.dataframe(X_train)
		st.sidebar.text(f"Training data size: {X_train.shape}")

		k = top.slider(f"Value of K ",1,100)
		
		cluster_names = top.text_input("Enter valid {} cluster seed names:".format(k))
		cluster_names= cluster_names.strip().split()
		TB = top.button("Train")
		
		if TB:
			clustered_data = KMC.k_means_clustering(X_train,cluster_names,label)
			middle.dataframe(clustered_data)
			clusters = clustered_data["cluster"].unique()
			middle.write(f"{k} distinct clusters are : {clusters}")
			plot(clustered_data,None,"cluster",algo_type)
			st.balloons()
		