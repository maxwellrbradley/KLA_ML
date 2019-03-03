import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plotter


import warnings  
warnings.filterwarnings('ignore')
def csvReader(file):
    return pd.read_csv(file, sep=",") # open file with , splitter


def algorithmSelector(X_train, y_train, X_test, y_test):
    # take data set, run data sets through each and return the classfier with the best confidence level
    classifiers = []
    maxConfidence = -1
    indexToReturn = -1
    
    for num, k in enumerate(['linear','poly','rbf','sigmoid']): # enumerates over all possible types of models
        clf = svm.SVR(kernel=k, gamma='auto')
        clf.fit(X_train, y_train) # train model on X_test and y_test
        confidence = clf.score(X_test, y_test) # check confidence score
        print(f"Confidence: {confidence} with model: {k}") # prints the confidence and model
        classifiers.append(clf) # sets classifier to the back of the array
        if confidence > maxConfidence: # if the confidence level is higher for this model
            maxConfidence = confidence # set max confidence to check which index to return 
            indexToReturn = num # sets the index to return to be the one in the array

    return classifiers[indexToReturn]
            

main_df = csvReader('~/Documents/Work/Internships/KLA_PE/data/farmerData.csv')
main_df.fillna(0, inplace=True)
main_df['Average Watered Per Week(gallons)'] = 0




################################################################################################
# GET INFORMATION ON LABEL
################################################################################################

#print(main_df.info())
#print(main_df['Weekly Yield (lbs of berries)'].describe())


################################################################################################
# STEP 0: AVERAGE OUT THE AMOUNT OF WATER USED PER WEEK
################################################################################################
totalWatered = 0
for index in range(0, len(main_df)):
    totalWatered += main_df.at[index,'Watering(gallons)']
    if ((index + 1) % 52) == 0: # if on the last part of a row
        for i in range(0, 52):
            main_df['Average Watered Per Week(gallons)'][index - i] = totalWatered / 52 # average out the amount watered
        totalWatered = 0


# set up total watered amount of gallons to be the average watered per week to be the rain fall and the average watering
#main_df['Total Watered(gallons)'] = main_df['Rain(gallons)'] + main_df['Average Watered Per Week(gallons)'] 

################################################################################################
# PLOT CORRELATION GRAPHS
################################################################################################
#plot = sb.distplot(main_df['Weekly Yield (lbs of berries)'],  color='blue', bins=int(100/5))
#plotter.xlabel('Yield of Berries')
#plotter.ylabel('Amount of samples?')
#plotter.show()
## good for saving distribution of data
#fig = plot.get_figure()
#fig.savefig('distribution.png')

################################################################################################
# CREATE CORRELATION GRAPH
################################################################################################
#correlation = main_df.corr()
#f, ax = plotter.subplots(figsize=(12,9))
#sb.heatmap(correlation, vmax=0.8, square=True)
#plotter.show()

#main_df = main_df.drop(columns = ['Watering(gallons)']) # drops unimportant watering in gallons
#main_df = main_df.drop(columns = ['Average Watered Per Week(gallons)']) # drops watering per week
#main_df = main_df.drop(columns = ['Rain(gallons)']) # drops the incoming rain
k = 12   # we'll look at the 10 largest correlations
correlation = main_df.corr()
cols = correlation.nlargest(k, 'Weekly Yield (lbs of berries)')['Weekly Yield (lbs of berries)'].index

print('The', k, 'columns with the highest correlation with Sale Price:')
print(list(cols))   # the k variables that correlate most highly with SalePrice

cm = np.corrcoef(main_df[cols].values.T)
f , ax = plotter.subplots(figsize = (20, 15))

plot = sb.heatmap(cm, 
            vmax = .9, 
            linewidths = 0.01, 
            square = True, 
            annot = True, 
            cmap = 'viridis',
            linecolor = "white", 
            xticklabels = cols.values, 
            annot_kws = {'size': 12}, 
            yticklabels = cols.values)
#plotter.show()
fig = plot.get_figure()
fig.savefig('correlation.png')

sb.set()
columns = ['Weekly Yield (lbs of berries)', 'Moisture %', 'Sun/TempF', 'Fertilizer (cubic_feet)', 'Watering(gallons)', 'Nutrition Lvl %', 'Fertilizer added cubic feet)', 'Mulch added (inches)', 'Mulch (inches)', 'Rain(gallons)']
plot = sb.pairplot(main_df[columns],size=20, kind='scatter', diag_kind='kde') #Had to change 'height' to 'size' because of the update in seaborn

fig = plot.get_figure()
fig.savefig('scatter.png')


#################################################################################################
## STEP 1: CREATE MAX YIELD MODEL WITH MOISTURE %, NUTRITION %, AND TEMPERATURE
#################################################################################################
#X = np.array(main_df[['Nutrition Lvl %', 'Moisture %', 'Sun/TempF']]) # set up input array to be all parts except for the points of berries
#y = np.array(main_df['Weekly Yield (lbs of berries)']) # set up output array to be the pounds of berries
#X = preprocessing.scale(X) # scale the input array to make it easier to use
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1) # puts results of testing and training into the X_train, y_train, X_test, and y_test vars. The test size is set up as 0.2, and the 
#mainClassifier = algorithmSelector(X_train, y_train, X_test, y_test)
#confidence = mainClassifier.score(X_test, y_test) # get confidence of overall test
#print(f"confidence of overall yield: {confidence}")
#
#
#################################################################################################
## STEP 2: CREATE RELATIONSHIP BETWEEN WATER, RAIN, AND MULCH TO MOISTURE PERCENTAGE
#################################################################################################
#
## need to keep mulch in there
## what is the correct relationship?
# 
#X = np.array(main_df[['Total Watered(gallons)', 'Mulch (inches)', 'Sun/TempF']]) # set up input array to be all parts except for the points of berries
#y = np.array(main_df['Moisture %']) # set up output array to be the pounds of berries
#X = preprocessing.scale(X) # scale the input array to make it easier to use
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1) # puts results of testing and training into the X_train, y_train, X_test, and y_test vars. The test size is set up as 0.2, and the 
#moistureClassifier = algorithmSelector(X_train, y_train, X_test, y_test)
#confidence = moistureClassifier.score(X_test, y_test) # get confidence of overall test
#print(f"confidence of water yield: {confidence}")

################################################################################################
# STEP 3: CREATE RELATIONSHIP BETWEEN NUTRITION LEVELS AND MULCH TO NUTURITION PERCENTAGE
################################################################################################



#x_input = np.array([97, 82, 73]) # feeds a new array into the model and runs the classifier on it
#x_input = np.array([0, 0, 0]) # feeds a new array into the model and runs the classifier on it

#x_input = main_df.loc[78, ['Nutrition Lvl %', 'Moisture %', 'Sun/TempF']]
#print(f"x_input before processing:\n{x_input}")
#x_input = preprocessing.scale(x_input)
#x_input = np.array(x_input)
##print(x_input)
#x_input = x_input.reshape(1, -1) # reshapes the array because it's a single sample
#
#print(clf.predict(x_input)) # print prediction

# sets up dataframe to contain the values that we're looking for

#clf = LinearRegression(n_jobs=-1) # set up linear regression training
#clf.fit(X_train, y_train) # train the regression classifier based on the X and y input data
#confidence = clf.score(X_test,y_test) # check to see how well the model does on the testing data
#print(f'Confidence of linear regression: {confidence}')

#forecast_col = 'Adj. Close' # set the forcasted column to be this one
#forecast_out = int(math.ceil(0.01 * len(df))) # forcast out ten days # figure out how far out to forcast the data
#print(f"Amount of days to predict: {forecast_out}")
#df['Price in Thirty-Five days'] = df[forecast_col].shift(-forecast_out) # creates a new column with the results of the forcast column shifted up by the forcast var 
#df.dropna(inplace=True) # drop n/a data sets
#X = np.array(df.drop(['Price in Thirty-Five days'], 1)) # copies the dataframe except for the price in ten days, which is the dependant var
#print(f"X Pre-scaling: {X}")
#X = preprocessing.scale(X) # scale the input array to make it easier to use
#y = np.array(df['Price in Thirty-Five days'])  # makes a new dataframe that only contains the price in ten days from the last one
#
#X_lately = X[-forecast_out:] # sets the lately values to be the values at the very end, where the price isn't calculated yet. In this case, it's from the end of the array back forecast out amount, to the very end, so the time from 35 days before now up until now
#print(f"X lately values: {X_lately}")
#X = X[:-forecast_out] # sets values from 0 to the end of the array minus forecast out
#y = y[:-forecast_out] # sets values from 0 to the end of the array minus forecast out
#print(f"X: {X}")
#
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2) # puts results of testing and training into the X_train, y_train, X_test, and y_test vars. The test size is set up as 0.2, and the train size is 0.8
#clf = LinearRegression(n_jobs=-1) # set up linear regression training
#
#clf.fit(X_train, y_train) # train the regression classifier based on the X and y input data
#
#confidence = clf.score(X_test,y_test) # check to see how well the model does on the testing data
#forecast_set = clf.predict(X_lately) # get the new set of X values to train on
#print(f"Forecast_set: \n{forecast_set}")
#
#
#
#def csvReader(file):
#    return pandas.read_csv(file, sep=",") # open file with , splitter




# for running with multiple models
#for k in ['linear','poly','rbf','sigmoid']:
#    clf = svm.SVR(kernel=k, gamma='auto')
#    clf.fit(X_train, y_train)
#    confidence = clf.score(X_test, y_test)
#    print(k,confidence)
#    confidenceArray.append((k, confidence))

# for reshpaing arrays of lengh 1
#x_input = x_input.reshape(1, -1) # reshapes the array because it's a single sample
#X = np.array(main_df['Total Watered(gallons)'])# set up input array to be all parts except for the points of berries
#X = X.reshape(-1, 1)