from os import replace
from matplotlib import axis
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Read in data from specified file and return and formats the relevent parts
returns the data as a pandas data frame
"""
def read_data(file):
    label_select = lambda x: x == 30 or 2 <= x <= 13 #predicate that selects columns 2-13 and field for alcohol usage
    
    df = pd.read_csv(file, names=range(2,33), usecols=label_select) #names start from 2 because 1st col automatically excluded

    labels= ["Trait"+str(i+1) for i in range(12)]+["alcohol"]
    
    #["age","gender","education","country","ethnicity","nscore","escore",
    #"oscore","ascore","cscore","impulsive","ss","alcohol"]
    df.columns = labels
    return df

def plot(x,y):
    fontsize = "30";
    params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (8, 8),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
    plt.rcParams.update(params)

    fig = plt.figure()
    ax = fig.add_subplot(111) 

    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    ax.scatter(x, y,color='red', alpha=.8, s=140, marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    #plt.savefig('plot_scatter2D.png')
    plt.show()

def plotv(x,y):
    fontsize = "30";
    params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (8, 8),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
    plt.rcParams.update(params)

    fig = plt.figure()
    ax = fig.add_subplot(111) 

    from scipy.spatial import KDTree
    tree = KDTree(x+y)

    # Best to warn the user; this can get slow
    print('Calculating a million values. This will take a while...')

    # We will sample both X1 and X2 dimensions and evaluate the model
    # at each X1,X2 coordinate in turn. Our model is simply the Y value
    # of the nearest neighbour
    
    # Define the resolution of the graph. Then choose 'res' points in
    # each dimension, linearly spaced between min and max
    res = 30
    xspace = np.linspace(0, 80, res)
    yspace = np.linspace(0, 100, res)

    # Create a grid of these two sequences
    xx, yy = np.meshgrid(xspace, yspace)

    # Finally, obtain a list of coordinate pairs
    xy = np.c_[xx.ravel(), yy.ravel()]

    # For each X1,X2 pair, find the nearest neighbour among our input data
    t = tree.query(xy)[1].reshape(res,res)
    tt = t.reshape(-1);             # Convert to 1D to make the loop easier...

    # For each X1,X2 pair, find the value corresponding to that nearest neighbour
    for index, val in enumerate(tt):
        tt[index] = y[tt[index]]

    tt = tt.reshape(res,res)        # Convert to 2D again

    # Now plot these values as a heatmap 
    plt.pcolor(xspace, yspace, tt, cmap='jet')

    # then add the original datapoints on top as reference
    #ax.scatter(x[:,0], x[:,1], s=140, c='w')
    plt.colorbar()

    # Label the axes to make the plot understandable
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Adjust the axis limits to remove unimportant areas
    ax.set_xlim(0,80)
    ax.set_ylim(0,100)

    # Save as an image file
    #plt.savefig('plot_voronoi.png')

    # Display in a window
    plt.show()

def max_index(arr):
    if len(arr) == 0: return -1
    v=arr[0]
    index=0
    for i in range(1,len(arr)):
        if v < arr[i]:
            v = arr[i]
            index = i
    return index

def split_countries(data,mapping,field):
    

    frame = dict()
    for m in mapping:
        title=mapping[m]
        data.insert(1,title, np.isclose(data[field],m))
        data[title]=data[title].replace({False:0,True:1})
    data.drop(field,1)
def main(t=12):

    
    data = read_data("Practical1/data/drug_consumption.data")


    

    mapping = {-0.09765 :"australia" 
    ,0.24923 :"canada" 
    ,-0.46841: "nz" 
    ,-0.28519 :"other" 
    ,0.21128 :"roi" 
    ,0.96082 :"uk" 
    ,-0.57009 :"us" }

    mapping = { -2.43591 : "Left school before 16 years"
    ,-1.73790 : "Left school at 16 years"
    ,-1.43719 : "Left school at 17 years"
    ,-1.22751 : "Left school at 18 years"
    ,-0.61113 : "Some college or university, no certificate or degree"
    ,-0.05921 : "Professional certificate/ diploma"
    ,0.45468 : "University degree"
    ,1.16365 : "Masters degree"
    ,1.98437 : "Doctorate degree" }

    #split_countries(data,mapping,"education")

    y=data.alcohol
    X = data.loc[:,"Trait1":"Trait"+str(t)]
    
    x_train, x_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.33, random_state=12)


    
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train,None)

    x_test = scale.fit_transform(x_test,None)

    ys=list()
    """for i in range(7):
        mapping = dict()
        for j in range(7):
            if i!=j:
                mapping["CL"+str(j)] = "other"#0 if i==j else 1
        ys.append(y_train.replace(mapping))"""

    
    
    pf = PolynomialFeatures(3,include_bias=False)
    
    print(len(pf.fit_transform(X)[0] ))
    print(len(X.columns))

    linreg = LinearRegression()
    #linreg.fit(x,y)
#
    #y_hat = linreg.predict(x)
    #print('MSE = ', mean_squared_error(y,y_hat))

    
    predictions = list()
    for i in range(1):
        logreg = LogisticRegression()
        
        logreg.fit(x_train,y_train)

        y_hat = logreg.predict(x_train)
        
        print('Accuracy = ', accuracy_score(y_train,y_hat,normalize=True))
        #y_hat = logreg.predict(x_test)

        #print('Accuracy = ', accuracy_score(y_test,y_hat,normalize=True))
    #for i in range(len(y_test)):
    #predictions = np.stack(predictions,axis=1)
    #y_hat = ["CL"+str(max_index(pred)) for pred in predictions]
    #print(y_hat)

    #print(y_hat)
    #plot(y,y_hat)
    

if __name__ == "__main__":
    for i in range(1,13):
        main(i)