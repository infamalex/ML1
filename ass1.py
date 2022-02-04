from os import replace
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

"""
Read in data from specified file and return and formats the relevent parts
returns the data as a pandas data frame
"""
def read_data(file):
    label_select = lambda x: x == 30 or 2 <= x <= 13 #predicate that selects columns 2-13 and 30
    df = pd.read_csv(file, names=range(2,33), usecols=label_select) #names start from 2 because 1st col automatically excluded

    df.columns = ["trait_" + str(i) for i in range(1,13)] + ["usage"]
    return df

def main():
    data = read_data("Practical1/data/drug_consumption.data")
    
    mapping = dict()
    for i in range(7):
        mapping["CL"+str(i)] = i

    x = data.loc[:,"trait_1":"trait_12"]
    y = data.usage.replace(mapping)
    
    linreg = LinearRegression()
    linreg.fit(x,y)

    y_hat = linreg.predict(x)
    print('MSE = ', mean_squared_error(y,y_hat))

    logreg = LogisticRegression(penalty= "l2", class_weight=None)
    logreg.fit(x,y)

    y_hat = logreg.predict(x)
    print('MSE2 = ', mean_squared_error(y,y_hat))

if __name__ == "__main__":
    main()