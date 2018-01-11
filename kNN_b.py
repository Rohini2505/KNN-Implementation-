#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 08:25:03 2017

@author: rmandge
"""

import pandas as pd
import numpy as np
import timeit

def normalizeDFtrain(dataFrame):
    dfNormalized = dataFrame.copy()
    colList = list(dataFrame.columns)
    #print(cols)
    
    for col in range(len(colList)):
        colMean = dataFrame[colList[col]].mean()
        colStd = dataFrame[colList[col]].std()
        #print(col,'= ', colMean)
        #print(col,'= ', colStd)
        dfNormalized[colList[col]] = (dataFrame[colList[col]] - colMean)/colStd
    
    return dfNormalized

def normalizeDFtest(traindataB4,testdataB4):
    dfNormalized = testdataB4.copy()
    colList = list(testdataB4.columns)
    #print(cols)
    
    for col in range(len(colList)):
        colMean = traindataB4[colList[col]].mean()
        colStd = traindataB4[colList[col]].std()
        #print(col,'= ', colMean)
        #print(col,'= ', colStd)
        dfNormalized[colList[col]] = (testdataB4[colList[col]] - colMean)/colStd
    
    return dfNormalized
    

def getDistance(testdata,traindata,trainlabel, testCount):
#    print("----- testdata", np.shape(testdata))
#    print("----- traindata", np.shape(traindata))
    distance_diff = traindata - testdata
    distance_squared = distance_diff**2
    sq_dist = distance_squared.sum(axis=1)
    dist = sq_dist**0.5
    dist.sort_values(axis=0, ascending=True, inplace=True)
    #distance = pd.DataFrame({"distance":dist})
    #print("----- distance", np.shape(distance))
    #tmptrainindex = pd.DataFrame({"trainindex":traindata.index})
    #tmptestlabel = pd.DataFrame({"testindex":[testCount]*np.shape(distance.values)[0]})
    
    #distance_df = pd.concat([distance, tmptrainindex,tmptestlabel], axis=1)#, ignore_index=True)
    tmptestindex = [testCount]*np.shape(dist.values)[0]
    #print(testRowIndexList)
    
    tempDistanceDict = { 'trainindex' : np.array(dist.index) , 'distance': dist.values, 'testindex' : tmptestindex}
    distance_df = pd.DataFrame(tempDistanceDict)

#    ind = distance_df.index
#    val = distance_df.index.values
#    print(ind)
#    print(val)
#    print("length of distance_df", len(distance_df))
#    print("Distance DF \n",distance_df)
    #sorted_dist = distance_df.sort_values(by='distance')
    return distance_df


#============ MAIN ==================
s = timeit.default_timer()    

traindatatmp = pd.read_csv("spam_train.csv")
testdatatmp = pd.read_csv("spam_test.csv")
traindataB4 = traindatatmp.loc[:,'f1':'f57']
testdataB4 = testdatatmp.loc[:,'f1':'f57']

traindata = normalizeDFtrain(traindataB4)
testdata = normalizeDFtest(traindataB4,testdataB4)
#print(type(testdata))
#print(traindata.head())
##print(testdata.loc[:,'Label'])
#print(testdata.head())


#trainlabel = traindatatmp.loc[:,'class']
#testlabel = testdatatmp.loc[:,'Label']
trainlabel = traindatatmp[['class']].copy()
testlabel = testdatatmp[['Label']].copy()
#print(testdatatmp.head())
#print(testlabel.index.values)
#df = testdatatmp.loc[testdatatmp['Label'] == 1]
#
#print("DF", df)

k=[1,5,11,21,41,61,81,101,201,401]
#k=[5,11]

testall = pd.DataFrame()
testall = testall.fillna(0)
i = 0
for testCount in testdata.itertuples(index=False, name='Pandas'):
    spam = 0
    nospam = 0
    #top_knn = getDistance(testdata,traindata,trainlabel,i)
    top_knn = getDistance(testCount,traindata,trainlabel,i)
#    print(top_knn)
    testall = testall.append(top_knn)
    i += 1
#    if(i > 5):
#            break
#print(testall)
#print("testall.shape",testall.shape)
print("\n")
print("kNN Algorithm - test Accuracies with Z-score Normalization features \n ")
for kcount in range(len(k)):
#    print("k[kcount]",k[kcount])
#    print("k[kcount]-1",k[kcount]-1)
    testRowCount = 0
    PredictLabel =[]
    for i1 in testdata.itertuples(index=False, name = 'Pandas'):
        
        distanceForRow = testall[testall['testindex']==testRowCount]
#        print(np.shape(distanceForRow))
#        print(distanceForRow)
        
#        print("Distancerow top testindex",distanceForRow.loc[:(k[kcount]),'testindex'])
#        print("Distancerow top trainindex",distanceForRow.loc[:(k[kcount]),'trainindex'])
        
        NNIndex = distanceForRow.loc[:(k[kcount]-1),'trainindex']
#        print(np.shape(distanceForRow.loc[:(k[kcount]-1),'trainindex']))
#        print("K count -- ",(k[kcount]-1))
#        NNIndex = distanceForRow.head(k(kcount))
#        print(np.shape(NNIndex))
#        print(NNIndex)
#        print("trainlabel ---- " , trainlabel)
        NNtrainLabel = trainlabel.iloc[NNIndex]['class'].value_counts()
#        print("NNtrainLabel" , NNtrainLabel)
        PredictLabel.append(NNtrainLabel.idxmax())
        testRowCount +=1
#        if(testRowCount > 5):
#            break
    
    tmpList = {'Label' : PredictLabel}
    PredictLabelDF = pd.DataFrame(tmpList)

    labelDiff = testlabel.sub(PredictLabelDF , axis=1)
    #print(differenceLabel)
    accurateCount = len(labelDiff[ labelDiff['Label'] ==0 ])
    
#    print('accurateCount ---- ', accurateCount)
    accuracyPercent = accurateCount/testlabel['Label'].count()*100
    print("AccuracyPercent for ",k[kcount], " is ",accuracyPercent)
##for kcount in range
#    for i in top_knn.iteritems():
#        print("i value",i)
#        if i == testlabel[testCount]:
#            print("1")
#        else:
#            print("0")
        
#for i in top_knn.iteritems():
#    print("i value",i)
#    if i == testlabel[0]:
#        print("1")
#    else:
#        print("0")
##print("testdatatmp \n",testdatatmp)

e = timeit.default_timer()        
print ("Execution time ",e - s)