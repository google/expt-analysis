#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# author: Reza Hosseini

""" This is an example of using the code to sequence the data"""

import sys
import os
import numpy as np
import pandas as pd
import random
import time
import datetime
import numpy as np
import time as time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
import re as re
import inspect
import scipy as scipy
import functools
import itertools
import operator
import warnings
import json
import IPython
import hashlib
import base64


def GetContent(fn):
  with open(fn, 'r') as f:
    content = f.read()

  return content


## specify the path for the source code
path = ''

srcFns = [path + 'expt-analysis/python/data_analysis.py',
          path + 'expt-analysis/python/sequential_data.py',
          path + 'expt-analysis/python/sequences_statistics_v1.py',
          path + 'expt-analysis/python/sequences_statistics_v2.py',
          path + 'expt-analysis/python/unit_analysis.py']

for fn in srcFns: exec(GetContent(fn=fn))


# Define a location for SQL Tables Log File (SQL tables are optional)
# and a writePath for the seq data files
# make sure these paths do exist
sqlTablesLogFile = '~/data/seq_data/seq_data_info.csv'
writePath = '~/data/seq_data/'

## define a figs path
figsPath = '~/data/seq_data/figs/'

## define a tables path for writing results tables
tablesPath = '~/data/seq_data/tables/'

# Example with simulated data demo purpose:

## step 1: simulate usage data
df = Sim_depUsageData(userNum=200, subSeqLen=4, repeatPattern=None)

## step 2: sequence the data
dataNamesList = ['test']
dataDesc = 'seq'
fnSuff = '.csv'
# trim is the length of sequences we are considering for finding significance
trim = 3
condDict = None

out = WriteSeqTable_forSql(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date', 'country'],
    seqPropCols=['form_factor'],
    timeGapMin=5,
    timeCol='time',
    timeColEnd='end_time',
    trim=trim,
    countDistinctCols=['user_id', 'seq_id'],
    condDict=None,
    addSeqPropMixedCols=['form_factor'],
    ordered=True,
    writePath=writePath,
    dataNamesList=dataNamesList,
    dataDesc=dataDesc,
    fnSuff=fnSuff,
    defineSqlTab=False,
    DefineSqlTable_fromFile=DefineSqlTable_fromFile,
    ExecSqlQuery=ExecSqlQuery,
    sqlTablePrefix="",
    timeGapDict=None,
    writeTableLogFn=sqlTablesLogFile)

# run this if you have implemented SQL query execution
# and like to use SQL tables rather than files
sqlStr = out['sqlStr']
Mark(sqlStr, color='purple', bold=True)
#ExecSqlQuery(sqlStr)


## look at the info table
for fn in srcFns: exec(GetContent(fn=fn))
seqTablesDf = ReadCsv(fn=sqlTablesLogFile)
Mark(
    seqTablesDf,
    text='set of available sql tables for finding sequences',
    color='purple',
    bold=True)


## step 3: get that particular table we need using the info table
rowNum = 0
row = seqTablesDf.iloc[rowNum]
trim = row['trim']
Mark(trim, 'trim')
seqPropCols = []

if str(row['seqPropCols']) != 'nan':
  seqPropCols = row['seqPropCols'].split(';')
  Mark(seqPropCols, 'seqPropCols are as follows:')

seqPropCols = (
    [x + '_parallel' for x in seqPropCols] +
    [x + '_mix' for x in seqPropCols])

countDistinctCols = []
if str(row['countDistinctCols']) != 'nan':
  countDistinctCols = row['countDistinctCols'].split(';')
  Mark(countDistinctCols, 'countDistinctCols are as follows:')

Mark(seqPropCols)
sqlTableName = row['sqlTableName']
fileName = row["writePath"] + row["fileName"] + ".csv"
Mark(sqlTableName, 'This is the sql table name you requested.')


## if want to load data through file
seqDf = ReadCsv(fileName)
Mark(seqDf.shape, 'data size (seqDf.shape):')
Mark(seqDf[:2], 'example seq data:')


## if want to load data via SQL (assuming the SQL functions are implemented)
# seqDf2 = ReadSqlTable(table=sqlTableName)
# Mark(seqDf2.shape, 'data size (seqDf.shape):')
# Mark(seqDf2[:2], 'example seq data:')


## step 4: finding the sig sequences which satisfy particular conditions
sliceCols = ['country']
auxSliceCols = ['trimmed_form_factor_parallel', 'trimmed_form_factor_parallel_mix']

## calculate significance
seqDfWithSignif = AddSeqProbCiMetrics(
    seqDf=seqDf.copy(),
    trim=int(trim),
    addCounts=True,
    sliceCols=sliceCols,
    auxSliceCols=auxSliceCols,
    seqCol='seq',
    countDistinctCols=countDistinctCols,
    seqCountMin=3)

## also calculate penetration:
# need to pass penetItemCols to do that
seqDfWithSignif2 = FindSigSeq_withPenet(
    seqDf=seqDf.copy(),
    trim=int(trim),
    seqCol='seq',
    sliceCols=sliceCols,
    auxSliceCols=auxSliceCols,
    countDistinctCols=countDistinctCols,
    penetItemCols=['user_id', 'date'],
    seqCountMin=3)

condDict = {
    #'country':['JP', 'US', 'FR']
    #'trimmed_form_factor_parallel_mix':['COMP']
}

## a set of values to be regex
regDict = {}


plt.figure()
Mark(text="SIG PLOTS + PENETRATION PLOT", color='blue', bold=True)


sigDict = Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif2.copy(),
    sliceCols=sliceCols,
    condDict=condDict,
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.05,
    orderByCol='relative_prob2',
    addPenetPlots=True,
    seqNumLimit = None,
    rotation=0,
    logScale=True,
    figSize=[8, 8],
    saveFig=True,
    figPath=figsPath,
    figFnPrefix=sqlTableName.replace('.', '_'),
    figFnExt='png',
    Open=OpenFile)

sigDf = sigDict['df']

sigDf = sigDict['df']
if (sigDf is not None):
  Mark(x=sigDf.shape, text="sigDf.shape:", color="green", bold=True)
  Mark(x=sigDf[:6], text="sigDf snapshot:", color="blue", bold=True)
  Write_sigSeqDf(
      sigDf=sigDf,
      sqlTableName=sqlTableName,
      path=tablesPath,
      regDict=regDict,
      condDict=condDict)
else:
  Mark(text='no data was found', color='red')
