#Gaussian Naive Bayes Classification
#Dataset: Iris Flower dataset  having three classes : Iris - setosa , Iris - Veriscolor , Iris - verginica


"""It is a Gaussian Naive Bayes Classification. First of all, it assumes that all the features in tha gien data are 
	normally distributed means possess Gaussian distribution. Hence if the the data has this property then accuracy 
	would be 100 % , otherwise less than 100 %."""
	
import numpy as np
import pandas as pd



def load_dataset(filename):
    df=pd.read_csv(filename)
    ##print(df.head(5))
    Y = list(df.columns)
    last = Y[-1]
    ##print(last)
    #randomly shuffle the data which is for test_data
    df = df.sample(frac=1).reset_index(drop=True)

    np.random.seed(2)
    r_no=np.random.randint(df.shape[0]-20)
    test_data= df[r_no:r_no+20]
    train_data=df.drop(range(r_no,r_no+20))
    
    classes=list(train_data[last].unique())
    
    return train_data,test_data,classes,last

def compute_mean_deviation(classes,data,last):
    
    mean_std={}
    #print("Classes are:	",classes)
    for i,c in enumerate(classes):
        samples=data[(data[last]==classes[i])]
        samples=samples.drop([last],axis=1)
        mean_std["mean"+str(i+1)]=samples.mean()
        mean_std["std"+str(i+1)]=samples.std()
        
    
    return mean_std

def split_input_output(data,last):
    out = data[last]
    inp = data.drop([last],axis=1)
    
    return inp,out

def compute_marginal_probs(classes,Y):
    assert(Y.shape==(len(Y),))
    counts=list(Y.value_counts())
    total=len(Y)
    marginal_prob={}
    for i,c in enumerate(counts):
        marginal_prob["class"+str(i+1)]=c/total

    return marginal_prob

def compute_conditional_probs(X,mean_std,c):
    
    assert(type(c)==int)
    
    m = np.array(mean_std["mean"+str(c)])
    s = np.array(mean_std["std"+str(c)])
    
    M = m.reshape(1,len(m))
    S = s.reshape(1,len(s))
    
    #probability distribution formula for gaussian distribution
    denom = S*np.sqrt(2*np.pi)
    power = np.power(np.divide(X-M,S),2)
    numer = np.exp(np.multiply(-0.5,power))
    prob = np.divide(numer,denom)
    
    #prob is multiplication of the conditional probalities of all the features
    prob = np.prod(prob,axis=1)
    
    
                       
    return prob


def gaussian_naive_bayes_classiefier(data,classes,last):
    mean_std=compute_mean_deviation(classes,data,last)
    marginal_prob = compute_marginal_probs(classes,data[last])
    return marginal_prob,mean_std


def predict(X,Y,mean_std,marginal_prob,classes):
    
    c=len(classes)
    assert(type(c)==int)
    
    Pred = np.empty((c,X.shape[0]))
    for i in range(1,c+1):
        C = marginal_prob["class"+str(i)]
        P = C*compute_conditional_probs(X,mean_std,i)
        Pred[i-1]=P
        
    #gives the index of maximum value
    pred_class = np.argmax(Pred.T,axis=1)
    
    predictions= []
    for i in range(len(pred_class)):
        predictions.append(classes[pred_class[i]])
    
    ##print(pred_class)
    '''for i in range(X.shape[0]):
        #print("{}".format(X.iloc[i]))
        #print("Predicted Class: {}   Actual Class: {}".format(predictions[i],Y_test.iloc[i]))
        #print()
        #print()'''
     
    correct_pred = list(predictions==Y)
    ##print(correct_pred)
    correct_count =0
    for c in correct_pred:
        if c:
            correct_count+=1

    accuracy = (correct_count)/len(pred_class)
    #print("Accuracy is : {:%}".format(accuracy))
    return predictions,accuracy

def naive_bayes_model(dataset):

    train_data,test_data,classes,last = load_dataset(dataset)

    #print("Shape of train_data: ",train_data.shape)
    #print("Shape of test_data: ",test_data.shape)

    marginal_prob,mean_std=gaussian_naive_bayes_classiefier(train_data,classes,last)
    ##print(mean_std["mean1"])
    X_train,Y_train = split_input_output(train_data,last)

    X_test,Y_test = split_input_output(test_data,last)
    #print()
    #print("\nTraining",end=" ")
    train_predict,_ = predict(X_train,Y_train,mean_std,marginal_prob,classes)
    #print("Testing",end=" ")
    test_predict,_ = predict(X_test,Y_test,mean_std,marginal_prob,classes)
    #print("\n\n")

    test_data["Prediction for "+last] = test_predict
    #print(test_data)
    #print("\n-------------------------------- successfully completed----------------------------\n\n")

def main(dataset):
    naive_bayes_model(dataset)

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
    
    
'''Accuarcy for : Iris => training 90% , testing 100%
	Accuarcy for : heart => training 83% , testing 85%
	Accuarcy for : wine=> training 73% , testing 75% '''

