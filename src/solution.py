################################################################################################
# OVERALL PROGRAM FLOW AND DESIGN CHOICES
################################################################################################

# read the farmer data in
# look at the overall stats on the incoming data
# get information on the label (Weekly Yield of berries)
    # 'label' is a machine learning term that is the output that you're looking to predict
    # a 'feature' is the converse of the label, the data that you are using to predict the label
# plot a distribution graph of the incoming label (yield)
# create a correlation heatmap to see the relationship between the label and other vars
    # this is to figure out whether any variables should be dropped because their 
    # correlations are too high to hold any valuable information independantly
    # we will use this data to create an overall prediction model
# create a correlation heatmap between moisture and all variables except for the label
    # this is because moisture is a strong influencer of the label according to the
    # correlation chart, and because it is not directly controlled, it needs to predicted
    # we will use this data to create a moisture prediction model
# create a scatter plot for more data visualization
###############################################################################################
# now that the graphs have been created, begin the number crunching
###############################################################################################
# create a yield prediction model using only moisture and sun/temp inputs
    # this is because these two variables correlated strongly with the yield and adding 
    # other variables into the mix decreased the R^2 cofficent (our measure of accuracy)
# create a moisture prediction model with only mulch and the watering in gallons
    # like the yield prediction model, only these features were used because they correlated 
    # strongly with the moisture percentage
# read in the output data provided by Robert that contained the forecast for the coming year
# optimize yield
    # for all the weeks in the forecast
        # change the amount of water
            # predict the amount of moisture that is associated with the mulch from the last week
            # predict the yield associated with that moisture and sun/temp
            # if the yield is higher than the previously predicted yields, save that amount of water
        # change the amount of mulch
            # predict the amount of moisture that is associated with that mulch and the water from the last step
            # predict the yield associated with that moisture and sun/temp
            # if the yield is higher than the previously predicted yields, save that amount of mulch
        
        # save the most yielding water and most yielding mulch to the output spreadsheets
   # save the output data to the spreadsheet

###############################################################################################
# assumptions made
###############################################################################################

# 0.15 inches of mulch are lost a week. This number was found by looking at the rate of decrease of mulch
# from the incoming data

###############################################################################################
# notes about running code
###############################################################################################

# this code was run in a virtual environment running python3.7
# to make this program work for another computer, the paths to the input and output
# files will need to be adjusted

###############################################################################################
# BEGIN SOURCE CODE
###############################################################################################

# import data frame library
import pandas as pd
# import numpy lib for data processing
import numpy as np 
# NOTE: sklearn is a python machine learning library
# import preprocessing and support vector machine that contains 
# machine learning functions
from sklearn import preprocessing, svm
# import data splitting
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate 
# import graphing library
import seaborn as sb
# import python plotting lib
import matplotlib
# import lib to show plots in realtime
import matplotlib.pyplot as plotter
# shut off warnings because constantly on in machine learning
import warnings  
warnings.filterwarnings('ignore')
# set the precision out the output to 2 decimal places
pd.set_option('precision', 2)
# set up the decrement rate of mulch to be 0.15, found from data
MULCH_DECREMENT_RATE = 0.15

################################################################################################
# HELPER FUNCTIONS
################################################################################################

# create function to read in csv file
def csvReader(file):
    # open file with , splitter
    return pd.read_csv(file, sep=",") 

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
        # train model on X_test and y_test
        clf.fit(X_train, y_train) 
        # check R^2 value, measure of accuracy
        confidence = clf.score(X_test, y_test) 
        # sets classifier to the back of the array
        classifiers.append(clf)
         # if the confidence level is higher for this model
        if confidence > maxConfidence:
            # set max confidence to check which index to return 
            maxConfidence = confidence 
            # sets the index to return the most accurate model
            indexToReturn = num 

    return classifiers[indexToReturn]

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
fig.savefig('distribution.png') # saves figure

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
# CREATE CORRELATION GRAPH BETWEEN MOISTURE % AND ALL OTHER VARIABLES
################################################################################################

# look at the 7 largest correlations
k = 7  
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
## STEP 1: CREATE MODEL FOR PREDICTING YIELD OF BERRIES WITH HIGH ACCURACY
################################################################################################# 

# set up the input to the training and testing to be only the moisture % and the sun/tempF 
# because of their high correlations
print(f"Array before overall prediction: \n{main_df.drop(columns = ['Weekly Yield (lbs of berries)', 'Nutrition %', 'Watering (gallons)', 'Fertilizer (cubic feet)', 'Mulch (inches)', 'Rain (gallons)'])}")
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
## STEP 2: CREATE MOISTURE PREDICTION MODEL
#################################################################################################

print(f"Array before overall prediction: \n{main_df[['Mulch (inches)','Watering (gallons)']]}")
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
## STEP 3: READ IN OUTPUT DATASET WITH SUN/TEMPF, RAINFALL, CURRENT MULCH, CURRENT FERT
# MULCH TO ADD, FERT TO ADD
################################################################################################

# read in the weather data sheet
output_df = csvReader('~/Documents/Work/Internships/KLA_PE/data/weather.csv')
# initialize data set with cleared values and set up as a float array
output_df['Current Mulch'] = 0
output_df['Current Mulch'] = output_df['Current Mulch'].astype('float64')
output_df['Watering (gallons)'] = 0
output_df['Watering (gallons)'] = output_df['Watering (gallons)'].astype('float64')
output_df['Mulch added (inches)'] = 0 
output_df['Mulch added (inches)'] = output_df['Mulch added (inches)'].astype('float64')
output_df['Fertilizer (cubic feet)'] = 0
output_df['Fertilizer (cubic feet)']= output_df['Fertilizer (cubic feet)'].astype('float64')
output_df['Fertilizer added (cubic feet)'] = 0
output_df['Fertilizer added (cubic feet)'] = output_df['Fertilizer added (cubic feet)'].astype('float64')
print(output_df.info())


################################################################################################
## STEP 4: OPTIMIZE YIELD
################################################################################################

# for each week in the total forcast
for week in range(0, len(output_df)):
    print(f'\n\nWeek {week}')
    # if it's the first week or the current amount of mulch < 0,
    # set the amount of mulch to be 0
    if week == 0 or output_df.loc[week-1, 'Current Mulch'] - MULCH_DECREMENT_RATE <= 0: # on first iteration
        currentMulch = 0
        prevMulch = 0
    else:
    # otherwise, set the amount of mulch to be the current amount - 0.15 according to empirical data
        prevMulch = output_df.loc[week-1, 'Current Mulch']
        currentMulch = prevMulch - MULCH_DECREMENT_RATE
    # get the current amount of sun coming in for the current week
    currentSun = output_df.loc[week, 'Sun/Temp Forecast (F)']
    # set up the max yield of the week to be zero
    maxYield = 0
    #BEGIN WATER PREDICTION
    bestWater = 0
    for water in range(3, 30):
        # take in mulch and water
        # create a moisture array with the amount of water as the independant varible
        # and the mulch held constant
        moistureArray = np.array([currentMulch, water]).reshape(1, -1)
        # predict the amount of moisture for the watering and mulch
        predictedMoisture = moisturePredictor.predict(moistureArray)
        # set up the array to predict yield with 
        yieldArray = np.array([predictedMoisture, currentSun]).reshape(1, -1)
        # get current yield using the overall predictor
        currentYield = mainPredictor.predict(yieldArray) 
        # if the yield predicted is greater than the max yield, set the max to that value
        if currentYield > maxYield:
            # save the water value for futre trials
            bestWater = water
            maxYield = currentYield
    # BEGIN MULCH PREDICTION
    # reset max yield
    maxYield = 0 
    bestMulch = 0
    mulchToAdd = 0
    while mulchToAdd < 8:
        # create array to predict on with current mulch and the best water
        moistureArray = np.array([currentMulch + mulchToAdd, bestWater]).reshape(1, -1)
        # predict the moisture with the mulch value and the water from the water prediction
        predictedMoisture = moisturePredictor.predict(moistureArray)
        # set up the yield array with the predicted moisture
        yieldArray = np.array([predictedMoisture, currentSun]).reshape(1, -1)
        # predict the yield for that moisture
        currentYield = mainPredictor.predict(yieldArray) 
        # if the yield predicted is greater than the max yield, set the max to that value
        # mark that amount of mulch to add to be ideal
        if currentYield > maxYield:
            bestMulch = mulchToAdd
            maxYield = currentYield
        
        # add to the ideal amount of mulch to add
        mulchToAdd += 1 
    # when the loop completes, fill in the output data frame

    output_df['Mulch added (inches)'][week] = bestMulch
    output_df['Current Mulch'][week] = currentMulch + bestMulch
    output_df['Watering (gallons)'][week] = bestWater

################################################################################################
## STEP 5: OUTPUT TO SPREADSHEET
################################################################################################

output_df.to_csv(r'data/output.csv')
