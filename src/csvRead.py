import csv
import pandas


#with open('~/Documents/Work/Internships/KLA_PE/data/farmerData.csv', 'r') as csv_file:
# need to run the file from the src dir
#csv_file= open('../data/farmerData.csv', 'r')
#csv_file= open('~/Documents/Work/Internships/KLA_PE/data/farmerData.csv', 'r')
#csv_reader = csv.reader(csv_file, delimiter=',')
#line_count = 0
#for row in csv_reader:
#    if line_count == 0:
#        print('Column names are {", ".join(row)}')
#        line_count += 1
#    else:
#        print(row)
#        line_count += 1



#df = pandas.read_csv('~/Documents/Work/Internships/KLA_PE/data/farmerData.csv', sep=",", names=['Fertilizer added (cubic feet)','Fertilizer (cubic feet)','Mulch added (inches)','Mulch (inches)','Rain(gallons)','Watering(gallons)','Nutrition Lvl %','Moisture %','Sun/TempF','Weekly Yield (lbs of berries)'])
df = pandas.read_csv('~/Documents/Work/Internships/KLA_PE/data/farmerData.csv', sep=",",)

print(df.shape)
print(df.iloc[0:19,0:])
