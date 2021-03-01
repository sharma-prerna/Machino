#deep neural network implementation
#Dataset1: cat vs non cat classification
#Dataset2: cat vs dog classification
#Dataset3: heart desease possibility prediction
#Datset4: wine quality classification


from os import listdir
from os.path import isfile,join
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

def create_boolean_dataset(n):

    data_range = np.array(range(0,2**n),dtype=int)
    train_input = np.empty((np.power(2,n),n),dtype=int)
    train_output = np.ones((1,2**n),dtype=int)
    train_output[:,0]= train_output[:,-1]=0

    for num in data_range:
        train_input[num]= binary_conversion(num,n)
    
    cols = range(1,n+1)
    train_input = pd.DataFrame(train_input,columns = cols)
    ##print(train_input,train_output)
    return train_input,train_output,train_input,train_output
    
def binary_conversion(num,n):
    binary_str = "{:08b}".format(num)
    binary_num = list(binary_str[-n:])
    return binary_num

#cat vs dog classification
def load_catdog_dataset():
    dataset="catdog_dataset2.hdf5"
    with h5py.File(dataset,"r") as f:
        ##print(f.keys())
        train=f['train_data']
        X_train=np.array(train["X_train_catdog"])
        Y_train=np.array(train["Y_train_catdog"])
        
        test=f['test_data']
        X_test=np.array(test["X_test_catdog"])
        Y_test=np.array(test["Y_test_catdog"])
        
    return X_train,Y_train,X_test,Y_test

def load_cat_noncat_dataset():
    f1="test_catvnoncat.h5"
    f2="train_catvnoncat.h5"
    
    #X_test=Y_test=X_train=Y_train=np.empty(1)
    
    with h5py.File(f1,"r") as test:
        X_test=np.array(test["test_set_x"][:])
        Y_test=np.array(test["test_set_y"][:])
        test_classes=test["list_classes"][:]
    
    with h5py.File(f2,"r") as train:
        X_train=np.array(train["train_set_x"][:])
        Y_train=np.array(train["train_set_y"][:])
        train_classes=train["list_classes"][:]
        ##print(train.keys())
        
    return X_train,Y_train,X_test,Y_test

def load_wine_dataset(filename):
    df=pd.read_csv(filename)
    np.random.seed(5)
    random_no=np.random.randint(df.shape[0]-50)
    test_data= df[random_no:random_no+50]
    train_data=df.drop(range(random_no,random_no+50))

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
    
    np.random.shuffle(np.array(df))
    np.random.seed(4)

    random_no=np.random.randint(df.shape[0]-50)
    test_data= df[random_no:random_no+50]
    train_data=df.drop(range(random_no,random_no+50))

    train_output=np.array(train_data['target'])
    test_output=np.array(test_data['target'])

    train_input=train_data.drop(['target'],axis=1)
    test_input=test_data.drop(['target'],axis=1)

    train_output=train_output.reshape(1,len(train_output))
    test_output=test_output.reshape(1,len(test_output))

    return train_input,train_output,test_input,test_output

#standardize the data
def normalize_IQR(X_train,X_test):
    train_IQR=X_train.quantile(q=0.75,axis=0)-X_train.quantile(q=0.25,axis=0)
    X_train_normalized=(X_train-X_train.quantile(q=0.25,axis=0))/train_IQR

    test_IQR=X_test.quantile(q=0.75,axis=0)-X_test.quantile(q=0.25,axis=0)
    X_test_normalized=(X_test-X_test.quantile(q=0.25,axis=0))/test_IQR
    return X_train_normalized,X_test_normalized


def normalize_data(X_train,X_test):
    train_range=X_train.max()-X_train.min()
    X_train_normalized=(X_train-X_train.min())/train_range
    
    test_range=X_test.max()-X_test.min()
    X_test_normalized=(X_test-X_test.min())/test_range
    
    return X_train_normalized,X_test_normalized


def initialize_parameters(layers):
    np.random.seed(2)
    parameters={}
    m=len(layers)
    for l in range(1,m):
        parameters["w"+str(l)]=np.random.randn(layers[l],layers[l-1])/(np.sqrt(layers[l-1])) #0.01
        parameters["b"+str(l)]=np.zeros((layers[l],1))

    return parameters

def sigmoid(x):
    return 1/np.array(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def tanh(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    s=sigmoid(x)
    return np.multiply(s,1-s)

def relu_derivative(x):
    x[x<=0]=0
    x[x>0]=1
    return x

def tanh_derivative(x):
    t=tanh(x)
    return (1-np.power(t,2))

def forward_propagation(A_prev,w,b,activation):
    ##print("{} * {} + {}".format(w.shape,A_prev.shape,b.shape))
    Z=np.dot(w,A_prev)+b
    if activation=="sigmoid":
        A=sigmoid(Z)
    elif activation=="relu":
        A=relu(Z)
    elif activation=="tanh":
        A=tanh(Z)
    else:
        #print("Invalid activation function")
        exit()
        
    assert(Z.shape==(w.shape[0],A_prev.shape[1]))
    assert(Z.shape==A.shape)
    
    forward_backup=(A_prev,w,b)
    backward_backup=Z
    ##print(forward_backup)
    ##print(backward_backup)
    return A,forward_backup,backward_backup

def backward_propagation(dA,forward_backup,backward_backup,activation):
    n=dA.shape[1]
    ##print("no of sample are: ",n)
    Z=backward_backup
    ##print("{} * {}".format(dA.shape,Z.shape))

    if activation=="sigmoid":
        G=sigmoid_derivative(Z)
        dZ=np.multiply(dA,G)
    elif activation=="relu":
        G=relu_derivative(Z)
        dZ=np.multiply(dA,G)
    elif activation=="tanh":
        G=tanh_derivative(Z)
        dZ=np.multiply(dA,G)
    else:
        #print("Invalid activation function")
        exit()
        
    assert(dZ.shape==dA.shape)
    
    A_prev,w,b=forward_backup
    dw=np.dot(dZ,A_prev.T)/n
    db=np.sum(dZ,axis=1,keepdims=True)/n
    dA_prev=np.dot(w.T,dZ)
    
    assert(dw.shape==w.shape)
    assert(db.shape==b.shape)
    assert(dA_prev.shape==(w.shape[1],n))
    
    return dA_prev,dw,db

def forward_propagation_model(X,parameters,input_activation,output_activation):
    m=len(parameters)//2
    ##print("no fo layers: ",m)
    forward_backups={}
    backward_backups={}
    A=X
    for l in range(1,m):
        ##print("{} * {} + {}".format(self.params["w"+str(l)].shape,A_prev.shape,self.params["b"+str(l)].shape))
        A_prev=A
        A,forward,backward=forward_propagation(A_prev,parameters["w"+str(l)],parameters["b"+str(l)],input_activation)
        forward_backups[str(l)]=forward
        backward_backups[str(l)]=backward  
    
    A_prev=A
    A,forward,backward=forward_propagation(A_prev,parameters["w"+str(m)],parameters["b"+str(m)],output_activation)
    forward_backups[str(m)]=forward
    backward_backups[str(m)]=backward
    
    assert(A.shape==(1,X.shape[1]))
    
    return A,forward_backups,backward_backups

def cost(A,Y):
    n=A.shape[1]
    ##print("no of samples: ",n)
    Y=Y.reshape(A.shape)
    c= -np.sum(np.multiply(Y,np.log(A))+np.multiply(1-Y,np.log(1-A)))/n
    c=np.squeeze(c)
    assert(c.shape==())
    return c

def backward_propagation_model(A,Y,forward_backups,backward_backups,input_activation,output_activation):
    
    m=len(forward_backups)
    ##print("no of layers: ",m)
    assert(A.shape==Y.shape)
    
    dA= np.divide(1-Y,1-A)-np.divide(Y,A)
    gradients={}
    ##print(dA)
    
    dA_prev,dw,db = backward_propagation(dA,forward_backups[str(m)],backward_backups[str(m)],output_activation)
    gradients["dw"+str(m)]=dw
    gradients["db"+str(m)]=db
    
    for l in reversed(range(1,m)):
        dA=dA_prev
        dA_prev,dw,db = backward_propagation(dA,forward_backups[str(l)],backward_backups[str(l)],input_activation)
        gradients["dw"+str(l)]=dw
        gradients["db"+str(l)]=db
        
    return gradients

def update_parameters(gradients,parameters,learning_rate):
    #self.m = number of layers
    m=len(parameters)//2
    ##print("no of layers:",m)
    for l in range(1,m+1):
        parameters["w"+str(l)]=parameters["w"+str(l)]-learning_rate*gradients["dw"+str(l)]
        parameters["b"+str(l)]=parameters["b"+str(l)]-learning_rate*gradients["db"+str(l)]
    return parameters


def deep_neural_network(layers,learning_rate,num_iterations,X,Y,input_activation="relu",output_activation="sigmoid"):
    
    parameters=initialize_parameters(layers)
    assert(parameters!=None)
    costs=[]
    for i in range(num_iterations):
        A,forward_backups,backward_backups=forward_propagation_model(X,parameters,input_activation,output_activation)
        
        if i%100==0:
            c=cost(A,Y)
            print("Costs after iteration {} {:.6f}".format(i,c))
        if i%50==0:
            c=cost(A,Y)
            costs.append(c)
            
        gradients=backward_propagation_model(A,Y,forward_backups,backward_backups,input_activation,output_activation)
        parameters=update_parameters(gradients,parameters,learning_rate*(num_iterations-i))
        
    return parameters,costs 

def predict(X,Y,parameters,input_activation,output_activation):
    A,_,_=forward_propagation_model(X,parameters,input_activation,output_activation)
    predictions=(A+1.5)//2
    Y=Y.reshape(predictions.shape)
    wrong=(Y-predictions)
    count=len(wrong[wrong!=0])
    acc = (Y.shape[1]-count)/Y.shape[1]
    #print("Accuracy : {:.2%}".format(acc))

    return predictions,acc,A

def print_prediction(test_data,Y_test,P_test,dataset,last):
    #printing the data

    Y_test = np.squeeze(Y_test)
    P_test = np.squeeze(P_test)
    ##print(Y_test,P_test)

    if dataset=="boolean":
        test_data["output"]= Y_test
        test_data["Prediction of "+last] = P_test.astype(int)
        #print(test_data)

    elif dataset=="wine":
        out = list(Y_test)
        pred = list(P_test)

        for i in range(len(out)):
            out[i]= "bad" if out[i]==0 else "good"

        for i in range(len(pred)):
            pred[i]= "bad" if pred[i]==0 else "good"
        
        test_data[last]=out
        test_data["Prediction of "+last] = pred
        #print(test_data)

    elif dataset=="heart":
        pred = list(P_test)
        out = list(Y_test)
        
        for i in range(len(out)):
            out[i]= "no" if out[i]==0 else "yes"

        for i in range(len(pred)):
            pred[i]= "no" if pred[i]==0 else "yes"

        test_data[last]=out
        test_data["Prediction of "+last] = pred
        #print(test_data)
    return test_data

def main(dataset,hyperparameters):

    layers,learning_rate,iterations = hyperparameters

    #deep neural network for can vs non cat classification
    if dataset=="catvnoncat":

        X_train_orig,Y_train_orig,X_test_orig,Y_test_orig=load_cat_noncat_dataset()

        test_data = X_test_orig
        last = "category"
        #flatten the data i.e change 12 ,12 , 3 array to  (1,12*12*3)
        m=X_train_orig.shape[0]
        n=X_test_orig.shape[0]
        X_train=X_train_orig.reshape(m,-1)
        X_test=X_test_orig.reshape(n,-1)
        Y_train=Y_train_orig.reshape(1,m)
        Y_test=Y_test_orig.reshape(1,n)

        #normalizing the array
        X_train=X_train/255
        X_test=X_test/255

    elif dataset=="wine":

        X_train,Y_train,X_test,Y_test=load_wine_dataset("wine.csv")

        test_data = X_test
        last = "quality"

        X_train,X_test=normalize_data(X_train,X_test)
        
    elif dataset=="heart":
        X_train,Y_train,X_test,Y_test=load_heart_dataset("heart.csv")

        test_data = X_test
        last = "target"

        X_train,X_test=normalize_data(X_train,X_test)

    #deep neural network for cat and dog classification
    elif dataset=="catdog":

        X_train,Y_train,X_test,Y_test=load_catdog_dataset()
        last ="category"
        test_data = X_test

        X_train = X_train.T
        X_test = X_test.T

        X_train=X_train/255
        X_test=X_test/255

    elif dataset=="boolean":
        #print("Enter no of inputs:",end=" ")
        try:
            n = int(input().strip())
        except:
            #print("Invalid entry ! Please enter any non negative integer between 1 to 7")
            #print("1 to 7 is for computation purpose")
            exit()

        if n<=0 or n>7:
            #print("Invalid entry ! Please enter any  non negative integer between 1 to 7")
            exit()

        X_train,Y_train,X_test,Y_test=create_boolean_dataset(n)
        last = "output"
        test_data = X_test
        layers =[n,2*n,1]
        #print(layers)
    else:
        #print("Invalid dataset")
        exit()

    #print("Training input and output shapes: ",X_train.shape,Y_train.shape)
    #print("Testing input and output shapes: ",X_test.shape,Y_test.shape)

    input_activation = "relu"
    output_activation = "sigmoid"
    parameters,_=deep_neural_network(layers,learning_rate,iterations,X_train.T,Y_train,input_activation,output_activation)
        
    #print("Training",end=" ")
    P_train,_,_ = predict(X_train.T,Y_train,parameters,input_activation,output_activation)
    #print("Testing",end=" ")
    P_test,_,_ = predict(X_test.T,Y_test,parameters,input_activation,output_activation)

    _=print_prediction(test_data,Y_test,P_test,dataset,last)

    #print("\n------------------------successfully completed------------------------\n\n")


if __name__=="__main__":

    #print("-----------Select one from the following list to train the model:----------------")
    #print("1 : XOR Gate learning")
    #print("2 : Can vs Non cat image classification")
    #print("3 : Wine Quality Prediction")
    #print("4 : Heart Disease Chances Prediction")
    #print("5 : Cat vs Dog Image classification")
    #print("Your input : ",end=" ")
    try:
        choice = int(input().strip())
    except:
        #print("Invalid input")
        exit()

    if choice==1:
        #print("\n--------------------- XOR Gate learning ------------------------------------")
        layers = None
        learning_rate=0.000015
        iterations=10001
        hyperparameters=(layers,learning_rate,iterations)
        main("boolean",hyperparameters)

    elif choice==2:
        #print("\n--------------------Can vs Non cat image classification--------------------")
        layers = [12288, 20, 7, 5, 1]
        learning_rate=0.000004
        iterations=2001
        hyperparameters = (layers,learning_rate,iterations)
        main("catvnoncat",hyperparameters)

    elif choice==3:
        #print("\n------------------------Wine Quality Prediction--------------------")
        layers=[11,1]
        learning_rate=0.00025
        iterations=2501
        hyperparameters = (layers,learning_rate,iterations)
        main("wine",hyperparameters)

    elif choice==4:
        #print("\n--------------------Heart Disease Chances Prediction----------------")
        layers = [13,1]
        learning_rate= 0.000055
        iterations = 2501
        hyperparameters = (layers,learning_rate,iterations)
        main("heart",hyperparameters)

    elif choice==5:
        #print("\n--------------------Cat vs Dog Image classification--------------------")
        layers = [12288, 20, 7, 4, 1]
        learning_rate=0.0000055
        iterations=3001
        hyperparameters = (layers,learning_rate,iterations)
        main("catdog",hyperparameters)

    else:
        #print("Invalid choice")
        exit()
#cd Documents\pdf\ML-prob-stat\Machine_learning_models

#for wine dataset : learning_rate/num_iterations=0.00025 , iterations = 2501 , layers=[11,1]
#training_accuracy is: 74%
#testing_accuracy is: 80%

#for cat vs noncat dataset : learning_rate/num_iterations=0.000004 ,iterations = 2001 layers = [12288, 20, 7, 5, 1]
#Training Accuracy is: 99.521531%
#Testing Accuracy is: 70.000000%

#for cat and dog dataset : learning_rate/num_iterations= 0.0000055 ,iterations = 3001 layers = [12288, 20, 7, 4, 1]
#training_accuracy is: 99%
#testing_accuracy is: 56%

#for heart disease dataset : learning_rate/num_iterations= 0.000055, iterations = 2501 layers = [13,1]
#training_accuracy is: 84%
#testing_accuracy is: 88%