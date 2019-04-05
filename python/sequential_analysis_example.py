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
          path + 'expt-analysis/python/sequential_data.py']

for fn in srcFns: exec(GetContent(fn=fn))



# upload your data or simulate data by running this:
# Simulating data for demo purpose: (analyst does this)
df = Sim_depUsageData(userNum=5, subSeqLen=4, repeatPattern=None)
## the simulated data is already sorted and has the correct timestamp
df['date'].value_counts()



"""
This function takes timestamped event data and create sequential data.

Inputs
df: data frame which has the data

timeCol: the column which include the event times

timeColEnd: the column which ends the end of the event, this could be passed
same as timeCol

seqDimCols: these are the building blocks for the sequence elements
for example  [form_factor, product]

partitionCols: these are partition columns used to partition the data.
you will be able to slice by them in the sequential data generated.
for example partitionCols = [user_id, country]

timeGap: the length of time gap (inactivity) used to break the sequences.

seqPropCols: columns which are properties of events to be also tracked.
we build parallel sequences to the main sequence using these properties.
for example if seqPropCols = []

seqPropColsDeduped: a subset of seqPropCols which are to be deduped as well

ordered: If this is True the code will assume the data is already ordered wrt
time. If not it will order the data.

Output:
output is a data frame which includes sequential data.
The sequences are denoted as  a1>a2>a3 where ">" is the separator

full_[col]_parallel: for a property given in col,
(we refer to these properties in code by seqPropCols),
this is the parallel sequence to “full_seq_deduped

full_seq_deduped:  this is the full sequence after complete deduping

full_seq_basket: this is the basket (set) of elements appearing
in the full sequence

trimmed_seq_deduped: this is the sequence after deduping and trimming.
This is usually the most important dimension for many use cases

trimmed_seq_basket: this is the set of elements appearing
in the trimmed sequence given in trimmed_seq_basket

trimmed_[col]_parallel: for a given property in col, e.g. form_factor,
this is the parallel sequence to the trimmed sequence

seq_shift_order: the data includes full sequences of actions for a user visit,
but it is also augmented by shifted version of sequences.
To restrict the data to sequences which start from time zero,
choose: seq_shift_order=0

full_seq_undeduped_length: the length of the undeduped sequence

full_seq_deduped_length: you can restrict the sequences of the represented data
by using this variable. For example you can choose all lengths
bigger than 1 to explore flows better.

event_1, event_2, …  You can restrict to for example second
event being equal to a particular event.

[col]_mix: if a sequence includes only one value for a property given in [col]
this will be equal to that values.
If the property includes multiple values during the sequence/journey
then its equal to “MIXED”. For example for col = [form_factor] we might have
a sequence which changes the form factor: COMP > PHONE > COMP
which will be assigned "MIXED"

[col]_parallel is the parallel sequence built along the main sequence to
track a specific property.

subseq_1_2, subseq_1_2_3, subseq_1_2_3_4: these are shorter versions of the
main sequence data given in "full_seq_deduped"
"""



# Generate the sequential data here from raw data
outputFileName = 'test_shifted_seq' #no suffix needed
timeCol = 'time'
# timeColEnd could be the same as timeCol if you don't have the end time
timeColEnd = 'end_time'
timeGap = 2*60
# make sure user_id column is a string column
df['user_id'] = df['user_id'].map(ShortHash)
partitionCols = ['user_id', 'country']
seqDimCols = ['prod', 'form_factor']
seqPropCols = ['prod', 'form_factor']
seqPropColsDeduped = seqPropCols
writePath = '~/work/tables/seq-data-analysis/'
trim = 3

seqDf = BuildAndWriteSeqDf(
  df=df,
  fn=outputFileName,
  seqDimCols=seqDimCols,
  partitionCols=partitionCols,
  timeGap=timeGap,
  trim=trim,
  timeCol=timeCol,
  timeColEnd=timeColEnd,
  seqPropCols=seqPropCols,
  seqPropColsDeduped=seqPropColsDeduped,
  writePath=writePath,
  addOrigSeqInfo=True,
  addBasket=True,
  addLagInfo=False,
  lagTrim=3,
  ordered=True,
  addResetDate_seqStartDate=True)

# inspect results
Mark(seqDf.shape)
seqDf[0:5]
