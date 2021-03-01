#MODEL: K nearest neighbour algorithm
#Dataset: Iris-data
#File Type: CSV

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Function for loading the data from the CSV file and cleaning it for further analysis
def load_dataset(filename):
    df=pd.read_csv(filename)
    ##print(df.head(5))
    Y = list(df.columns)
    last = Y[-1]
    ##print(last)
    #randomly shuffle the data which is for test_data
    np.random.seed(2)
    df = df.sample(frac=1).reset_index(drop=True)

    r_no=np.random.randint(df.shape[0]-20)
    test_data= df[r_no:r_no+20]
    train_data=df.drop(range(r_no,r_no+20))
    classes=list(train_data[last].unique())

    train_output=list(train_data[last])
    test_output=list(test_data[last])
    
    train_input=train_data.drop([last],axis=1)
    test_input=test_data.drop([last],axis=1)
    
    return train_input,train_output,test_input,test_output,classes,last

#Function for plotting scatter graph
def plot(data,test,classes,last):
    plt.style.use('seaborn')
    
    #classes = [Iris- setosa, Iris- versicolor, Iris-verginica]
    #plt.scatter(x=y=data,cmap='inferno')
    cols = list(data.columns)

    #class 1
    x1= np.array(data[data[last]==classes[0]][cols[0]])
    y1= np.array(data[data[last]==classes[0]][cols[1]])
    
    plt.scatter(x1,y1,color='g',label=classes[0],edgecolor='black')

    #class2
    x2= np.array(data[data[last]==classes[1]][cols[0]])
    y2= np.array(data[data[last]==classes[1]][cols[1]])
    
    plt.scatter(x2,y2,color='r',label=classes[1],edgecolor='black')

    if len(classes)>2:
        x3= data[data[last]==classes[2]][cols[0]]
        y3= data[data[last]==classes[2]][cols[1]]
        
        plt.scatter(x3,y3,color='b',label=classes[2],edgecolor='black')
    
    x4,y4 = test[[cols[0]]], test[[cols[1]]]
    plt.scatter(x4,y4,s =150,color='black',label='test instances',edgecolor='black',marker='*',alpha=0.9)
    plt.title('K nearest neighbours classification')
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.legend()
    plt.tight_layout()
    plt.show()

def euclid_distance(X,Y):
    dist= np.sqrt(np.sum(np.power(X-Y,2),axis=1))
    return dist


#Function for Finding k nearest neighbours
def k_nearest_neighbors(dist,cols,k):

    df = pd.DataFrame(data=dist,columns=cols,index=range(dist.shape[0]))
    #print("----------------------------------------------------------------")

    most_common_neigh = []
    for i in range(df.shape[0]):

        sorted_row = df.sort_values(by=i,axis=1,ascending=True)
        sorted_row = list(sorted_row.columns)
        nearest_neigh= sorted_row[:k]
        ##print("----------------------------Columns-------------------------------------",nearest_neigh)
        most_common_neigh.append(max(nearest_neigh,key=nearest_neigh.count))

    return most_common_neigh

#Accuracy check function
def accuracy(prediction,actual):
    count=0
    for p,a in zip(prediction,actual):
        if p==a:
            count+=1

    return (count/len(actual))

def predict_knn(X,Y,T,k):

    distances = np.empty((T.shape[0],X.shape[0]))
    Test = np.array(T)
    for i,t in enumerate(Test):
        dist = euclid_distance(X,t)
        distances[i] = dist

    prediction = k_nearest_neighbors(distances,Y,k)
    return prediction


def knn_model(dataset,k):

    X_train,Y_train,X_test,Y_test,classes,last = load_dataset(dataset)

    prediction = predict_knn(X_train,Y_train,X_test,k)
    acc=accuracy(prediction,Y_test)
    #print("Prediction Accuracy : {:%}".format(acc))

    
    X_test[last]=Y_test
    X_test["Prediction of "+last]=prediction
    #print(X_test)
    #print("\n-------------------------------- Successfully completed ----------------------------\n")

    X_train[last] =Y_train
    ##print(X_train)
    plot(X_train,X_test,classes,last)

def main(dataset):

    #print("Enter the value of k (neighbours) :",end=" ")
    try:
        k = int(input().strip())
        #print()

    except:
        #print("Please enter a positive integer number")
        exit()

    if k<=0:
        #print("Please enter a positive integer number")
        exit()

    knn_model(dataset,k)


if __name__=="__main__":
    #print("-----------Select one from the following list to train the model:----------------")
    #print("1 : Iris Flower Classification ")
    #print("2 : Wine Quality Classification")
    #print("3 : Heart Disease Chances Prediction")
    #print("Your input : ",end=" ")
    try:
        choice = int(input().strip())
    except:
        #print("Invalid input")
        exit()

    if choice==1:
        #print("\n------------------------ Iris Flower Classification --------------------------")
        main("Iris.csv")

    elif choice==2:
        #print("\n------------------------ Wine Quality Classification  --------------------------")
        main("wine.csv")

    elif choice==3:
        #print("\n-------------------------Heart Disease Chances Prediction-----------------------")
        main("heart.csv")

    else:
        #print("Invalid choice")
        exit()
        

#cd Documents\pdf\ML-prob-stat\Machine_learning_models

#Iris dataset: k=3 accuracy =100% for k=3,k=5
#wine quality dataset: 65% for k=3,k=5
#heart disease dataset: 60% for k=5