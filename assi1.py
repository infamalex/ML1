#!.venv/bin/python
from cmath import e
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

#ansi color codes
BLUE='\033[94m'
RED='\033[91m'
ENDC = '\033[0m'
UNDERLINE = '\033[4m'
YEL='\033[93m'

"""
Read in data from specified file and return and formats the relevent parts
returns the data as a pandas data frame
"""
def read_data(file,size,end_traits):
    #label_select = lambda p: p == y or x_start <= p < x_end #predicate that selects columns 2-13 and field for alcohol usage
    
    df = pd.read_csv(file, names=range(2,size+2)) #names start from 2 because 1st col automatically excluded
    labels= ["Trait"+str(i+1) for i in range(end_traits)]+\
            ["Class"+str(i+1) for i in range(size-end_traits)]
    
    df.columns = labels
    return df

"""
Train a logistic regression classifeir and 
"""
def test(X,Y,x_del,weight=None,penalty='none',C=1):
    x_train, x_valid, y_train, y_valid = \
    train_test_split(X, Y, test_size=0.25, random_state=45)

    logreg = LogisticRegression(class_weight=weight,penalty=penalty,C=C)
    logreg.fit(x_train,y_train)
    y_pred = logreg.predict(x_valid)
    return (y_pred, y_valid, logreg)

def get_rates(matrix):
    
    labels=list(matrix.keys())
    labels.sort()  
    counts = dict()
    for l in labels:
        counts[l]={"tp":0,"fp":0,"fn":0}
    for l in labels:
        counts[l]["tp"]=matrix[l][labels.index(l)]
        counts[l]["fp"]=sum(matrix[l])-counts[l]["tp"]#positves minues tp
        actual_class=[ matrix[l][labels.index(r)] for r in labels] #get column for actual class
        counts[l]["fp"]=sum(actual_class)-counts[l]["tp"]#counts for actual class minus tp
    return counts
        
"""Calculate the Precission and Recall of a """
def p_and_r(rates):
    print(YEL+"Precission and Recall")
    for l in rates:
        print(BLUE,l,"Precision ="+RED,rates[l]["tp"]/(rates[l]["tp"]+rates[l]["fp"]))
    print(ENDC,end="")

def accuracy(counts,total):
    return sum([counts[i]["tp"] for i in counts])/total   

"""
returns a dictionary containing the confusion matrix for predicted data.
Each entry in the dictionary contains the label
"""
def confusion_matrix(y_pred, y_true):
    labels=list(set(y_true)) #extract the classes from the data
    labels.sort()
    y_pred=list(y_pred)
    y_true=list(y_true)

    matrix = dict()
    for l in labels: #initialising the matrix with an array
        matrix[l]=[0]*len(labels) 
    for i in range(len(y_pred)):
    
        row = y_pred[i]
        col = labels.index(y_true[i])
        matrix[row][col]+=1
    return matrix

"""
Print my confusion matrix based on the format of 
"""
def print_matrix(matrix):
    labels=list(matrix.keys())
    labels.sort()

    print(YEL+"{:^{}}".format("y=prediction, x=true",(len(labels)+1)*6))
    #print top labels
    print(BLUE,end="")    
    for l in [""] + labels:
        print("{:>6}".format(l),end="")
    print(ENDC)

    #print side labels and data
    for l in labels:
        print(BLUE+"{:>6}".format(l),end="")
        for r in matrix[l]:
            print(RED+"{:>6}".format(r),end="")
        print(ENDC)

"""
Maps an existing collumn into a dummy variable
mapping:A dictionary 
data:Dataframe containing the data
"""
def split_column(data,mapping,field):
    frame = dict()
    for m in mapping:
        title=mapping[m]
        data.insert(1,title, np.isclose(data[field],m))
        data[title]=data[title].replace({False:0,True:1})
    data.drop(field,1)

def main():
    data = read_data("Practical1/data/drug_consumption.data",31,12) #select fields 2-13 and 30 from the data
    data = data[data.Class18=="CL0"] #filter responses that have allegedly taken Semer

    Y=data.Class17
    X = data.loc[:,"Trait1":"Trait12"]
    
    #split train and test data
    x_train, x_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.25, random_state=69)

    #Scale the data
    scale = Normalizer() #scale train and test data separately
    x_train = scale.fit_transform(x_train,None)
    x_test = scale.fit_transform(x_test,None)

    y_pred, y_true= test(x_train,y_train,x_test)[0:2]

    print('Basic Test:')
    print('Accuracy =', accuracy_score(y_true,y_pred))

    y_pred, y_true = test(x_train,y_train,x_test, weight="balanced")[0:2]
    print(YEL+'Balanced Test:')
    print(BLUE,'Accuracy ='+RED, accuracy_score(y_true,y_pred),ENDC)

    
    m=confusion_matrix(y_pred,y_true)
    counts = get_rates(m)
    p_and_r(counts)
    print(precision_score(y_true,y_pred,average=None))
    print(sum([counts[i]["tp"] for i in counts])/len(y_pred))

    #print confusion matrix
    print_matrix(m)
    print(confusion_matrix(y_true,y_pred))

    y_pred, y_true = test(x_train,y_train,x_train)[0:2]
    print(YEL+'Train Test:')

    #Accuracy of the 
    m=confusion_matrix(y_pred,y_true)
    counts = get_rates(m)
    print(BLUE,'Accuracy ='+RED,accuracy(counts,len(y_pred)) ,ENDC)
    
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(ENDC)
        raise e
