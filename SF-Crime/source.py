#SF-crime ML project
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:54:56 2016

@author: Mohit
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as knc
import csv 
import numpy as np
from sklearn.cross_validation import StratifiedKFold
#inputting data

'''
reading training and testing csv files
header = 0 since we know that 0 is header row
setting parse dates to true allows pandas to recognize them as dates
ref from http://goo.gl/z5AIBn
pandas doc @ http://goo.gl/uUOzqr
'''

print ( 'SAN FRANCISCO CRIME CLASSIFICATION')
print()
print("DATA INPUTTING...")

train = pd.read_csv('train.csv', header = 0, parse_dates = True, low_memory= False)
test = pd.read_csv('test.csv' , header =0, parse_dates = True, low_memory= False)

print()
print("DATA INPUTTING DONE...")

'''
scikit doc @ http://goo.gl/QPiytA
LE
crime category (categorical) data is floated
'''

print('Data before munging ')

print('Train data column values : ',  list(train.columns.values))
print(train.dtypes)
print('Test data column values : ',   list(test.columns.values))
print(test.dtypes)

#DATA munging
print()
print("WORKING ON TRAIN DATA ...")
print()

le = preprocessing.LabelEncoder()
le_category = preprocessing.LabelEncoder()

train = train.drop(['Dates','Descript' , 'Address', 'Resolution' ] , axis = 1)
train.Category = le_category.fit_transform(train.Category)
train.DayOfWeek = le.fit_transform(train.DayOfWeek)
train.PdDistrict = le.fit_transform(train.PdDistrict)

print("DONE WORKING ON TRAIN DATA ...")
print()

print("WORKING ON TEST DATA...")
print()

ids = test['Id'].values
test = test.drop(['Id', 'Dates', 'Address'] , axis = 1)

test['DayOfWeek'] = test['DayOfWeek'].astype(str)
test['PdDistrict'] = test['PdDistrict'].astype(str)

test.DayOfWeek = le.fit_transform(test.DayOfWeek)
test.PdDistrict = le.fit_transform(test.PdDistrict)

print("DONE WORKING ON TEST DATA...")
print()

print('Data munging is completed...')
print()

print('Data after munging ')
print()
print('Train data column values : ',  list(train.columns.values))
print()
print(train.dtypes)
print()
print('Test data column values : ',   list(test.columns.values))
print()
print(test.dtypes)

print()
print("CONVERTING TO NUMPY...")
print()

#converting to numpy array
train_data = np.array(train)
print()
test_data = np.array(test)

print("CONVERTED TO NUMPY ...")
print()

print("SPLITTING TRAIN DATA ...")
print()

features = []
target = []

#obtaining categories in target    
target = train['Category'].values

features = train_data[0:, 1:5]
model = knc(n_neighbors =  9)

#model = knc(n_neighbors = 9, n_jobs = -1)
#
print('TRAINING THE MODEL USING...', model)

print()
print('MODELLING DATA...')
#splitting into train and target variables
print()

model.fit(features,target)

print('MODELLING DONE...')
print()
print('PREDICTING TEST SET...')
print()
output = model.predict(test_data).astype(int)
print('WRITING TO RESULTS...')

'''
open func doc @
https://goo.gl/gUXa02
'''



#newline eliminated by using newline = ''
'''ref @ http://goo.gl/xZivIn

LE decoded using inverse_transform
doc @ http://goo.gl/iPblkL
'''

op_file = open("results.csv", "w", newline='')
file_object = csv.writer(op_file)
file_object.writerow(["Id","Category"])
file_object.writerows(zip(ids, le_category.inverse_transform(output)))
op_file.close()
#Converting output
df = pd.read_csv('results.csv', sep=',', encoding= 'utf-8', header= 0)

res_converter = pd.get_dummies(df.Category)
res_converter= res_converter.T.reindex()
print(res_converter)
df_transpose = res_converter.transpose()
df_transpose.to_csv('results.csv')

print('cross validation')
kf = StratifiedKFold(len(train_data), 10)

'''CROSS VALIDATION
num_folds = 10
subset_size = len(train_data)/num_folds
for i in range(num_folds):
    testing_this_round = train_data[i*subset_size:][1:5]
    training_this_round = train_data[2*i*subset_size:][1:5]
    print(training_this_round)
    print(testing_this_round)
CROSS VALIDATION'''
