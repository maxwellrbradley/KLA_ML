import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

# sets up dataframe to contain the values that we're looking for
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['Volitility'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100 # defines a new part of the dataframe calced from the columns
df['Percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df= df[['Adj. Close','Volitility', 'Percent_change', 'Adj. Volume']] # refines data frame to contain only the smaller set 

forecast_col = 'Adj. Close' # set the forcasted column to be this one
forecast_out = int(math.ceil(0.01 * len(df))) # forcast out ten days # figure out how far out to forcast the data
print(f"Amount of days to predict: {forecast_out}")
df['Price in Thirty-Five days'] = df[forecast_col].shift(-forecast_out) # creates a new column with the results of the forcast column shifted up by the forcast var 
df.dropna(inplace=True) # drop n/a data sets
X = np.array(df.drop(['Price in Thirty-Five days'], 1)) # copies the dataframe except for the price in ten days, which is the dependant var
print(f"X Pre-scaling: {X}")
X = preprocessing.scale(X) # scale the input array to make it easier to use
y = np.array(df['Price in Thirty-Five days'])  # makes a new dataframe that only contains the price in ten days from the last one

X_lately = X[-forecast_out:] # sets the lately values to be the values at the very end, where the price isn't calculated yet. In this case, it's from the end of the array back forecast out amount, to the very end, so the time from 35 days before now up until now
print(f"X lately values: {X_lately}")
X = X[:-forecast_out] # sets values from 0 to the end of the array minus forecast out
y = y[:-forecast_out] # sets values from 0 to the end of the array minus forecast out
print(f"X: {X}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2) # puts results of testing and training into the X_train, y_train, X_test, and y_test vars. The test size is set up as 0.2, and the train size is 0.8
clf = LinearRegression(n_jobs=-1) # set up linear regression training

clf.fit(X_train, y_train) # train the regression classifier based on the X and y input data

confidence = clf.score(X_test,y_test) # check to see how well the model does on the testing data
forecast_set = clf.predict(X_lately) # get the new set of X values to train on
print(f"Forecast_set: \n{forecast_set}")




