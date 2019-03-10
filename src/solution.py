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
pd.set_option('precision', 2)

MULCH_DECREMENT_RATE = 0.15
def csvReader(file):
    return pd.read_csv(file, sep=",") # open file with , splitter

# MACHINE LEARNING ALGORITHM
# run data through a particular regression kernel
# train the model on the data
# check to see accuracy of model
# return the regression classifer with the highest score

def algorithmSelector(X_train, y_train, X_test, y_test):
    # take data set, run data sets through each and return the classfier with the best confidence level
    classifiers = []
    maxConfidence = -1
    indexToReturn = -1
    
    for num, k in enumerate(['linear','poly','rbf','sigmoid']): # enumerates over all possible types of kernels
        clf = svm.SVR(kernel=k, gamma='auto')
        clf.fit(X_train, y_train) # train model on X_test and y_test
        confidence = clf.score(X_test, y_test) # check R^2 value, measure of accuracy
        print(f"Confidence: {confidence} with model: {k}") # prints the confidence and model
        classifiers.append(clf) # sets classifier to the back of the array
        if confidence > maxConfidence: # if the confidence level is higher for this model
            maxConfidence = confidence # set max confidence to check which index to return 
            indexToReturn = num # sets the index to return the most accurate model

    return classifiers[indexToReturn]


#SCRIPT BEGINS HERE
            
################################################################################################
# PROCESS INCOMING DATA
################################################################################################
# read in data from csv file
main_df = csvReader('~/Documents/Work/Internships/KLA_PE/data/farmerData.csv')
# fills parts of incoming data that don't have a value with 0 
main_df.fillna(0, inplace=True) 

################################################################################################
# GET INFORMATION ON LABEL (weekly yield of berries)
################################################################################################

print(main_df.info())
print(main_df['Weekly Yield (lbs of berries)'].describe())

################################################################################################
# PLOT DISTRIBUTION GRAPH FOR THE OVERALL YIELD
################################################################################################

# create a plot of the distribution of the output
plot = sb.distplot(main_df['Weekly Yield (lbs of berries)'],  color='blue', bins=int(100/5))
plotter.xlabel('Yield of Berries') # labels x axis 
plotter.ylabel('Samples distributed') # labels y axis
fig = plot.get_figure() # gets figure
fig.savefig('distribution.png')# saves figure

################################################################################################
# CREATE CORRELATION GRAPH BETWEEN OVERALL YIELD AND ALL OTHER VARIABLES
################################################################################################

# deletes mulch and fertizer added because already reflected in fertilizer amount and current mulch
main_df.drop(inplace=True, columns = ['Mulch added (inches)', 'Fertilizer added (cubic feet)'])
# get correlations between 8 most correlated columns
k = 8   
# create correlation of main dataframe
correlation = main_df.corr()
# get columns with 8 largest correlations
cols = correlation.nlargest(k, 'Weekly Yield (lbs of berries)')['Weekly Yield (lbs of berries)'].index
# get correlations between different variables
cm = np.corrcoef(main_df[cols].values.T) 
# gets suplots
f , ax = plotter.subplots(figsize = (20, 15))

plot = sb.heatmap(cm, 
            vmax = .9, # max color shown in heatmap
            linewidths = 0.01, # linewidth of graph
            square = True, 
            annot = True, 
            cmap = 'viridis', # color type
            linecolor = "white", 
            xticklabels = cols.values, # give x tick names of graphs
            annot_kws = {'size': 12}, # sizing
            yticklabels = cols.values) # set up ytick names same as x

# save correlation graph
fig = plot.get_figure()
fig.savefig('correlation.png') 

################################################################################################
# CREATE SCATTER PLOT
################################################################################################

# get columns to create scatter plot between
columns = ['Weekly Yield (lbs of berries)', 'Moisture %', 'Sun/TempF', 'Fertilizer (cubic feet)', 'Watering (gallons)', 'Nutrition %', 'Mulch (inches)', 'Rain (gallons)']
# create scatter plot
sb.set()
plot = sb.pairplot(main_df[columns],size=2, kind='scatter', vars=columns, diag_kind='kde') 
# save figure
plot.savefig('scatter.png')


#################################################################################################
## STEP 0: CREATE MODEL FOR PREDICTING YIELD OF BERRIES WITH HIGH ACCURACY
 ################################################################################################# 

# set up the input to the training and testing to be only the moisture % and the sun/tempF 
#because of their high correlations
X = np.array(main_df.drop(columns = ['Weekly Yield (lbs of berries)', 'Nutrition %', 'Watering (gallons)', 'Fertilizer (cubic feet)', 'Mulch (inches)', 'Rain (gallons)'])) 
# scale the input array to make processing faster
x = preprocessing.scale(X) 
# make the output array the thing we want to predict, aka the label
y = np.array(main_df['Weekly Yield (lbs of berries)']) 
# set up the arrays to test and train on by splitting up the main data frame into a data input and 
# output to train on, and then an input and output to test the data on
# split: 90% train, 10% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
# set the main overall predictor to be the output of the machine learning algorithm
# machine learning algorithm defined on line 25
mainPredictor = algorithmSelector(X_train, y_train, X_test, y_test)

################################################################################################
## STEP 1: CREATE MOISTURE PREDICTION MODEL
#################################################################################################
# look at the 8 largest correlations
k = 8  
# drop the label because not looking for correlation between label and moisture
correlation = main_df.drop(columns=['Weekly Yield (lbs of berries)']).corr()
cols = correlation.nlargest(k, 'Moisture %')['Moisture %'].index
# set up subplot
cm = np.corrcoef(main_df[cols].values.T)
f , ax = plotter.subplots(figsize = (20, 15))
# create correlation plot
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
# save the moisture correlation plot
fig = plot.get_figure()
fig.savefig('moistureCorrelation.png')
# next step: 
# check about relationship to moisture, and see what the biggest correlators are there
#print(f"main_df without cols for moisture: {main_df.drop(columns=['Weekly Yield (lbs of berries)', 'Moisture %', 'Sun/TempF', 'Nutrition %', 'Fertilizer (cubic feet)', 'Rain (gallons)'])}") 
# drop all columns except mulch in inches and watering gallons because of their high correlation with moisture %
X = np.array(main_df[['Mulch (inches)','Watering (gallons)']])
# set up output array to be the pounds of berries
y = np.array(main_df['Moisture %']) 
# scale the input array to make it easier to use
X = preprocessing.scale(X) 
# split up testing and training with 90/10 training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
# create moisture predictor based on the machine learning algorithm defined on line 25
moisturePredictor = algorithmSelector(X_train, y_train, X_test, y_test)

################################################################################################
# STEP 3: READ IN OUTPUT DATASET WITH SUN/TEMPF, RAINFALL, CURRENT MULCH, CURRENT FERT
# MULCH TO ADD, FERT TO ADD
################################################################################################
# read in the weather data sheet
output_df = csvReader('~/Documents/Work/Internships/KLA_PE/data/weather.csv')
# set up the current mulch of the whole system to be 0
output_df['Current Mulch'] = 0
# clear the watering of all weeks
output_df['Watering (gallons)'] = 0
# clear the yield of all weeks
output_df['Yield'] = 0 # will this be needed?
output_df['Mulch added (inches)'] = 0 # will this be needed?
output_df['Fertilizer (cubic feet)'] = 0 # will this be needed?
output_df['Fertilizer added (cubic feet)'] = 0 # will this be needed?
################################################################################################
# STEP 4: ENTER MAIN ALGORITHM
################################################################################################

# for each week in the total forcast
for week in range(0, len(output_df)):
    print(f'\n\nWeek {week}')
    # if it's the first week or the current amount of mulch < 0,
    # set the amount of mulch to be 0.4
    if week == 0 or output_df.loc[week-1, 'Current Mulch'] < 0.4: # on first iteration
        currentMulch = 0.4
    else:
    # otherwise, set the amount of mulch to be the current amount - 0.15 according to empirical data
        prevMulch = output_df.loc[week-1, 'Current Mulch']
        currentMulch = prevMulch - MULCH_DECREMENT_RATE
        print(f'mulch from last week : {prevMulch}')
    # get the current amount of sun coming in for the current week
    currentSun = output_df.loc[week, 'Sun/Temp Forecast (F)']
#    print(f"current mulch: {currentMulch}")
    # set up the max yield of the week to be zero
    maxYield = 0
#BEGIN WATER PREDICTION
    # variable to hold the value of the best amount of water to add
    bestWater = 0
    for water in range(3, 30):
        # take in mulch and water
        # create a moisture array with the amount of water as the independant varible
        # and the mulch held constant
        moistureArray = np.array([currentMulch, water]).reshape(1, -1)
        print(f'moistureArray from watering: {moistureArray}')
        predictedMoisture = moisturePredictor.predict(moistureArray)
        print(f"predictedMoisture: {predictedMoisture}") # LINE THAT IS CAUSING THE PROBLEM
        # assuming that moisture comes first rather than sun/tempF
        yieldArray = np.array([predictedMoisture, currentSun]).reshape(1, -1)
        currentYield = mainPredictor.predict(yieldArray) 
#        print(f'Current yield for water: {currentYield}\n')
        if currentYield > maxYield:
#            print("Water making max yield")
            bestWater = water
            maxYield = currentYield
#        print(maxYield)

# BEGIN MULCH PREDICTION
    maxYield = 0
    bestMulch = 0
    mulchToAdd = 0
    while mulchToAdd < 8:
        # create array to predict on with current mulch and the best water
        moistureArray = np.array([currentMulch + mulchToAdd, bestWater]).reshape(1, -1)
        print(f'Moisture array from mulch: {moistureArray}')
        predictedMoisture = moisturePredictor.predict(moistureArray)
        print(f"predictedMoisture: {predictedMoisture}") # LINE THAT IS CAUSING THE PROBLEM
        yieldArray = np.array([predictedMoisture, currentSun]).reshape(1, -1)
        currentYield = mainPredictor.predict(yieldArray) 
        # check if the yield is better with the addition of mulch
#        print(f'Current yield for mulch: {currentYield}')
        if currentYield > maxYield:
            bestMulch = mulchToAdd
            maxYield = currentYield
#            print(f'Mulch yield changing: {maxYield}')
        
        # add to the ideal amount of mulch to add
        mulchToAdd += 0.1 

    output_df['Mulch added (inches)'][week] = bestMulch
    print(f'Mulch added: {bestMulch}')
    print(f'current mulch at end: {currentMulch}')
    
    output_df['Current Mulch'][week] = currentMulch + bestMulch
    print(f"Array value: {output_df['Current Mulch'][week]}")
    output_df['Watering (gallons)'][week] = bestWater
    output_df['Yield'][week] = maxYield # will this be needed?



print(f"Final output:\n\n{output_df}")

output_df.to_csv(r'data/output.csv')


        # add 0.1 inches of mulch to the overall amount




#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at
#    input = np.array([output_df.at[i, 'Moisture %'], output_df.at[i, 'Sun/Temp Forecast (F)'], output_df.at[i, 'Watering (gallons)']])
#    print(f"Prediction with 75, 0, 0: {mainClassifier.predict(np.array([75,0, 0]).reshape(1, -1))}")
#    input = input.reshape(1,-1)
#    output_df.at[i, 'Yield'] = mainClassifier.predict(input)
    









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


# for reshpaing arrays of lengh 1
#x_input = x_input.reshape(1, -1) # reshapes the array because it's a single sample
#X = np.array(main_df['Total Watered(gallons)'])# set up input array to be all parts except for the points of berries
#X = X.reshape(-1, 1)
