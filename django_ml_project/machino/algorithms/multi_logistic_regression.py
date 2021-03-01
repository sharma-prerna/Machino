# multivariable logistic regression (binary classification)
#Dataset1: predict whether the wine is good or bad
#Dataset2: predict whether the person has heart disease or not
#File type: CSV

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
		

def create_boolean_dataset(n,choice):
	data_range = np.array(range(0,2**n),dtype=int)
	train_input = np.empty((np.power(2,n),n),dtype=int)

	for num in data_range:
		train_input[num]= binary_conversion(num,n)


	if choice=="OR":
		train_output = np.ones((1,2**n),dtype=int)
		train_output[:,0]= 0

	elif choice=="NAND":
		train_output = np.ones((1,2**n),dtype=int)
		train_output[:,-1] = 0

	elif choice=="NOR":
		train_output = np.zeros((1,2**n),dtype=int)
		train_output[:,0]= 1

	elif choice=="AND":
		train_output = np.zeros((1,2**n),dtype=int)
		train_output[:,-1] = 1

	else:
		train_output = np.array([[1,0]])
	
	cols = range(1,n+1)
	train_input = pd.DataFrame(train_input,columns = cols)
	##print(train_input,train_output)
	return train_input,train_output,train_input,train_output
	
def binary_conversion(num,n):
	binary_str = "{:08b}".format(num)
	binary_num = list(binary_str[8-n:])

	return binary_num

def load_wine_dataset(filename):
	df=pd.read_csv(filename)

	np.random.seed(5)
	random_no=np.random.randint(df.shape[0]-20)
	test_data= df[random_no:random_no+20]
	train_data=df.drop(range(random_no,random_no+20))

	train_output=train_data['quality']
	train_output=train_output.replace("bad",0)
	train_output=train_output.replace("good",1)
	train_input=train_data.drop(['quality'],axis=1)

	test_output=test_data['quality']
	test_output=test_output.replace("bad",0)
	test_output=test_output.replace("good",1)
	test_input=test_data.drop(['quality'],axis=1)

	test_output=np.array(test_output).reshape(1,len(test_output))
	train_output=np.array(train_output).reshape(1,len(train_output))

	return train_input,train_output,test_input,test_output

def load_heart_dataset(filename):
	df=pd.read_csv(filename)
	
	df = df.sample(frac=1).reset_index(drop=True)
	np.random.seed(4)

	random_no=np.random.randint(df.shape[0]-20)
	test_data= df[random_no:random_no+20]
	train_data=df.drop(range(random_no,random_no+20))

	train_output=np.array(train_data['target'])
	test_output=np.array(test_data['target'])

	train_input=train_data.drop(['target'],axis=1)
	test_input=test_data.drop(['target'],axis=1)

	train_output=train_output.reshape(1,len(train_output))
	test_output=test_output.reshape(1,len(test_output))
	return train_input,train_output,test_input,test_output

#this standardization results mean=0 and std=1 hence range of data would be -1 to +1
def standardize_data(X):
	X=(X-X.mean())/X.std()
	return X

#following normlization results range of data from 0 to 1
def normalize_data(X):
	mini=X.min()
	maxi=X.max()
	X=(X-mini)/(maxi-mini)
	return X

#interquartile ratio
def IQR(X):
	Q1=X.quantile(q=0.25,axis=0)
	Q1=X.quantile(q=0.75,axis=0)
	X=(X-Q1)/(Q3-Q1)
	return X

def sigmoid(x):
	return 1/np.array(1+np.exp(-x))

def sigmoid_derivative(x):
	s=sigmoid(x)
	return s*(1-s)

def initialize_parameters(features):
	w=np.zeros((1,features))
	b=0
	return w,b

def hypothesis(X,w,b):
	Z=np.dot(w,X)+b
	return Z

def activation(Z):
	return sigmoid(Z)

def cost(A,Y):
	assert(A.shape==Y.shape)
	n=A.shape[1]

	c= -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/n
	c=np.squeeze(c)

	assert(c.shape==())
	return c

def gradient_descent(X,Y,Z,A,w,b):
	#number of samples
	m=X.shape[1]

	Y=Y.reshape(A.shape)
	#this dL/dA where L is the cost function
	dA= -(np.divide(Y,A)-np.divide(1-Y,1-A))

	#computing dL/dZ=dL/dA * dA/dZ
	dZ=dA*sigmoid_derivative(Z)

	#computing dL/dW = dL/dZ * dZ/dW, Z=W*A_prev+b
	dw=np.dot(dZ,X.T)/m

	#computing dL/db=dZ*1
	db=np.sum(dZ,axis=1,keepdims=True)/m

	return dw,db

def update_parameters(dw,db,w,b,learning_rate):
	
	assert(dw.shape==w.shape)

	w= w-learning_rate*dw
	b= b-learning_rate*db

	return w,b

def predict(X,Y,w,b):

	Z=hypothesis(X,w,b)
	A=sigmoid(Z)
	P = (A+1.5)//2.0
	wrong_pred=np.abs(Y-P)
	wrong_pred=wrong_pred[wrong_pred!=0]
	accuracy=(Y.shape[1]-len(wrong_pred))/Y.shape[1]

	#print("accuracy is: {:.2%}".format(accuracy))
	P = np.squeeze(P)

	return P,accuracy,A


def logistic_regression(X,Y,learning_rate=0.05,num_iterations=1000):
	#number of features
	n=X.shape[0]

	#number of samples
	m=X.shape[1]

	w,b=initialize_parameters(n)
	costs=[]
	for i in range(num_iterations):

		Z=hypothesis(X,w,b)
		A=activation(Z)
		dw,db=gradient_descent(X,Y,Z,A,w,b)
		w,b=update_parameters(dw,db,w,b,learning_rate*(num_iterations-i))

		if not i%100:
			c=cost(A,Y)
			costs.append(c)
			#print("Cost after iterations {} : {:.6f}".format(i,c))

	#print("------------------- model trained ---------------------")
	return w,b,costs

def print_prediction(test_data,Y_test,P_test,dataset,last):
    #printing the data

    Y_test = np.squeeze(Y_test)
    P_test = np.squeeze(P_test)
 
    if dataset=="boolean":
        test_data["output"]= Y_test
        test_data["Prediction of "+last] = P_test.astype(int)

    elif dataset=="wine.csv":
        out = list(Y_test)
        pred = list(P_test)

        for i in range(len(out)):
            out[i]= "bad" if out[i]==0 else "good"

        for i in range(len(pred)):
            pred[i]= "bad" if pred[i]==0 else "good"
        
        test_data[last]=out
        test_data["Prediction of "+last] = pred

    elif dataset=="heart.csv":
        pred = list(P_test)
        out = list(Y_test)
        
        for i in range(len(out)):
            out[i]= "no" if out[i]==0 else "yes"

        for i in range(len(pred)):
            pred[i]= "no" if pred[i]==0 else "yes"

        test_data[last]=out
        test_data["Prediction of "+last] = pred

    return test_data

def main(dataset,hyperparameters):

	learning_rate,num_iterations = hyperparameters
	if dataset=="wine.csv":
		X_train,Y_train,X_test,Y_test = load_wine_dataset(dataset)
		last = "quality"
		test_data = X_test

	elif dataset=="heart.csv":
		X_train,Y_train,X_test,Y_test= load_heart_dataset(dataset)	
		last = "target"	
		test_data = X_test

	elif dataset=="boolean":
		#print("Enter no of inputs:",end=" ")
		try:
			n = int(input().strip())
		except:
			#print("Invalid entry ! Please enter any positive integer")
			exit()

		if n<=0:
			#print("Invalid entry ! Please enter any  non negative integer")
			exit()

		valid_choice = ["AND","OR","NAND","NOR","NOT"]
		#print("Pick your choice:  from {} :".format(valid_choice),end=" ")
		try:
			choice = input().strip().upper()
		except:
			#print("Invalid input")
			exit()
		
		if choice not in valid_choice:
			#print("Invalid choice")
			exit()

		X_train,Y_train,X_test,Y_test=create_boolean_dataset(n,choice)
		last = "output"
		test_data = X_test

		if n==1:
			X_test = np.array(X_test).reshape(2,1)
			X_train = np.array(X_test).reshape(2,1)

	else:
		#print("Oops ! Incorrect dataset")
		exit()

	#print("shape of training data: ",X_train.shape,Y_train.shape)
	#print("shape of training data: ",X_test.shape,Y_test.shape)

	X_train=normalize_data(X_train)
	X_test=normalize_data(X_test)

	w,b,_=logistic_regression(X_train.T,Y_train,learning_rate,num_iterations)

	#print("\nTraining",end=' ')
	P_train,_,_ = predict(X_train.T,Y_train,w,b)

	#print("Testing",end=' ')
	P_test,_,_ = predict(X_test.T,Y_test,w,b)
	#print("\n\n")
	
	data = print_prediction(test_data,Y_test,P_test,dataset,last)
	#print("\n------------------------successfully completed------------------------\n\n")

if __name__=="__main__":

	#print("-----------Select one from the following list to train the model:----------------")
	#print("1 : Logic gates learning")
	#print("2 : Wine Quality Prediction")
	#print("3 : Heart Disease Chances Prediction")
	#print("Your input : ",end=" ")
	try:
		choice = int(input().strip())
	except:
		#print("Invalid input")
		exit()

	if choice==1:
		#print("\n--------------------- Logic gates learning ------------------------------------")
		hyperparameters=(0.01,501)
		main("boolean",hyperparameters)
	elif choice==2:
		#print("\n-------------------- Wine Quality Prediction --------------------")
		hyperparameters =(0.0005,2001)
		main("wine.csv",hyperparameters)
	elif choice==3:
		#print("\n-------------------- Heart Disease Possibility Prediction --------------------")
		hyperparameters =(0.000055,2501)
		main("heart.csv",hyperparameters)
	else:
		#print("Invalid choice")
		exit()


#for logic gates : learning_rate/num_iterations= 0.01 ,iterations = 501
#training_accuracy is: 100
#testing_accuracy is: 100

#for wine dataset : learning_rate/num_iterations= 0.0005 ,iterations = 2001
#training_accuracy is: 74%
#testing_accuracy is: 80%

#for heart disease dataset : learning_rate/num_iterations= 0.000055 , iterations = 2501
#training_accuracy is: 84%
#testing_accuracy is: 90%