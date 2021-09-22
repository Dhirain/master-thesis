#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 10 00:34:01 2021
@author: dhirain
"""
print("Welcome to ML benchmark test")

print("In this test different ML model are test for CPU performance comparision")
print("")
import time
total_start = time.time()

def custom_prediction_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('odi.csv')
X = dataset.iloc[:,[7,8,9,12,13]].values
y = dataset.iloc[:, 14].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Test 1: Evaluating Linear regression")
#Fitting Linear regressor
from sklearn.linear_model import LinearRegression
lin = LinearRegression(n_jobs=4)
lin.fit(X_train,y_train)
# Testing the dataset on trained model for Linear Regression
y_pred = lin.predict(X_test)
score = lin.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test1time = time.time() - total_start
print("Time: "+ str(round(test1time,2))+"sec")


print("Test 2: Evaluating Decision Tress")
#fitting Decision tree
from sklearn.tree import DecisionTreeRegressor
reg_dec = DecisionTreeRegressor(max_depth=2)
reg_dec.fit(X_train, y_train)
# Testing the dataset on trained model for Decision Tree
y_pred = reg_dec.predict(X_test)
score = reg_dec.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test2time = time.time() - (total_start + test1time)
print("Time: " + str(round(test2time, 2))+"sec")

print("Test 3: Evaluating Random Forest")
#Fitting random forest
from sklearn.ensemble import RandomForestRegressor
reg_forest = RandomForestRegressor(n_estimators=100,max_features=None,n_jobs=4)
reg_forest.fit(X_train,y_train)
# Testing the dataset on trained model for Random Forest
y_pred = reg_forest.predict(X_test)
score = reg_forest.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test3time = time.time() - (total_start + test1time + test2time)
print("Time: " + str(round(test3time, 2))+"sec")


print("Test 4: Evaluating XGBClassifier")
from xgboost import XGBClassifier
xgboost_model = XGBClassifier(tree_method='hist',n_jobs=4)
#xgboost_model.fit(X_train, y_train)
# Testing the dataset on trained model for Random Forest
y_pred = xgboost_model.predict(X_test)
score = xgboost_model.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test4time = time.time() - (total_start + test1time + test2time + test3time)
print("Time: " + str(round(test4time, 2))+"sec")

print("Test 5: Evaluating XGBRegressor")
from xgboost import XGBRegressor
xgboost_model = XGBRegressor(tree_method='hist',n_jobs=4)
xgboost_model.fit(X_train, y_train)
# Testing the dataset on trained model for Random Forest
y_pred = xgboost_model.predict(X_test)
score = xgboost_model.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test5time = time.time() - (total_start + test1time + test2time + test3time + test4time)
print("Time: " + str(round(test5time, 2))+"sec")


print("Test 6: Evaluating PassiveAggressiveClassifier")
from sklearn.linear_model import PassiveAggressiveClassifier
pass_agg = PassiveAggressiveClassifier(n_jobs=4)
pass_agg.fit(X_train, y_train)
# Testing the dataset on trained model for Decision Tree
y_pred = pass_agg.predict(X_test)
score = pass_agg.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test6time = time.time() - (total_start + test1time + test2time + test3time + test4time + test5time )
print("Time: " + str(round(test6time, 2))+"sec")


print("Test 7: Evaluating SGDClassifier")
#fitting SVR
from sklearn.linear_model import SGDClassifier
reg_sgd = SGDClassifier(n_jobs=4)
reg_sgd.fit(X_train, y_train)
# Testing the dataset on trained model for Support vector
y_pred = reg_sgd.predict(X_test)
score = reg_sgd.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test7time = time.time() - (total_start + test1time + test2time + test3time + test4time + test5time + test6time )
print("Time: " + str(round(test7time, 2))+"sec")

print("Test 8: Evaluating Perceptron")
#fitting SVR
from sklearn.linear_model import Perceptron
prec = Perceptron(n_jobs=4)
prec.fit(X_train, y_train)
# Testing the dataset on trained model for Support vector
y_pred = reg_sgd.predict(X_test)
score = reg_sgd.score(X_test,y_test)*100
#print("R square: " , score)
#print("Custom accuracy: " , custom_prediction_accuracy(y_test,y_pred,20))
test8time = time.time() - (total_start + test1time + test2time + test3time + test4time + test5time + test6time + test7time)
print("Time: " + str(round(test8time, 2))+"sec")

total_end = time.time()
timetaken = round(total_end-total_start,2)
print('Total time taken in seconds: ' + str(timetaken) + ' sec')