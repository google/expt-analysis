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

""" Functions to to generate sequential data from timestamped data
and work with sequential data
we allow you to define what dimensions each sequence element contains
e.g. we can have a one-dimensional sequence of the form: browsingFeat>mailingFeat>watchFeat
or a 3-dim sequence of the form browsingFeat-COMP-VIEW>mailingFeat-PHN-READ
this code includes function for augmenting the data by shifting sequences
this code includes methods to add useful slicing methods to sequences when
generating them
for example columns to keep track of event_1st event_2nd etc in the sequences
this also includes functions to extract significant transitions in the sequence
as well as significant triples etc
"""


## simulate usage data with timestamps
## this is for testing code etc
def GenUsageData(
    userNum=None,
    userIds=None,
    dt1=datetime.datetime(2017, 4, 12),
    dt2=datetime.datetime(2017, 4, 12, 1, 0, 0),
    timeGridLen='1min',
    durLimit=None,
    prodsChoice=[
        'browsingFeat', 'mailingFeat',
        'editingFeat', 'exploreFeat', 'photoFeat',
        'PresFeat', 'locFeat', 'StorageFeat']):

  timeCol = 'time'
  timeColEnd = 'end_time'
  userCol = 'user_id'
  respCol = 'prod'
  timeGrid = pd.date_range(start=dt1, end=dt2, freq=timeGridLen)

  if userIds is None:
    userIds = pd.Series(range(userNum))

  userDf = pd.DataFrame({
    userCol: userIds,
    'country': np.random.choice(a=['US', 'JP', 'FR'],
                                size=userNum, replace=True)})
  timeDf = pd.DataFrame({timeCol: timeGrid})
  userDf['key'] = 0
  timeDf['key'] = 0
  df = userDf.merge(timeDf, how='outer', on=['key'])
  del df['key']
  size = df.shape[0]
  df[respCol] = np.random.choice(
      a=prodsChoice,
      size=size,
      replace=True)
  df['form_factor'] = np.random.choice(a=['COMP', 'PHN'],
                                       size=size, replace=True)
  df2 = df.sample(frac=0.5, replace=False)
  df2 = df2.sort_values(['user_id', 'time'])

  def F(df):
    df['delta'] = ((df[timeCol] - df[timeCol].shift()).fillna(0)).shift(-1).fillna(0)
    df['prop'] = np.random.uniform(low=1.0, high=5.0, size=df.shape[0])
    df['delta'] = (df['delta'] / df['prop'])
    df[timeColEnd] = df[timeCol] + df['delta']
    return(df)

  g = df2.groupby(['user_id'], as_index=False)
  df3 = g.apply(F)
  df3['date'] = df3['time'].dt.date
  df3[timeColEnd] = pd.DatetimeIndex(df3[timeColEnd]).round('1s')
  del df3['delta']
  del df3['prop']

  df3[userCol] = df3[userCol].map(str)
  df3['start_hour'] = (df3['time'].dt.hour).map(str)
  df3 = df3[[
      'country', 'user_id', 'date', 'prod', 'form_factor',
      'time', 'end_time', 'start_hour']]
  df3['dur_secs'] = (df3['end_time'] - df3['time']) / np.timedelta64(1, 's')
  if durLimit is not None:
    df3['dur_secs'] = df3['dur_secs'].map(lambda x: min(x, durLimit))

  return df3

'''
GenUsageData(5)
'''
## generate experiment data with patterns
def GenUsageData_withExpt(
    userIdsPair,
    dt1,
    dt2,
    timeGridLenPair=['2h', '2h'],
    durLimitPair = [3600, 3600],
    prodsChoicePair = [
        ['browsingFeat', 'mailingFeat', 'editingFeat',
         'exploreFeat', 'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat'],
        ['browsingFeat', 'mailingFeat', 'editingFeat',
         'exploreFeat', 'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat', 'browsingFeat',
         'editingFeat']]):

  dfList = []
  labels = ['base', 'test']

  for i in range(2):
    res = GenUsageData(
       userIds=userIdsPair[i],
       dt1=dt1,
       dt2=dt2,
       timeGridLen=timeGridLenPair[i],
       durLimit=durLimitPair[i],
       prodsChoice=prodsChoicePair[i])
    res['expt'] = labels[i]
    dfList.append(res)

  outDf = dfList[0].append(dfList[1], ignore_index=True)

  return outDf

'''
df = GenUsageData_withExpt(
    userIdsPair=[range(10), range(11, 20)],
    dt1=datetime.datetime(2017, 4, 12),
    dt2=datetime.datetime(2017, 4, 14),
    timeGridLenPair=['2h', '2h'],
    durLimitPair = [3600, 3000],
    prodsChoicePair = [
        ['browsingFeat', 'mailingFeat', 'editingFeat', 'exploreFeat',
         'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat'],
        ['browsingFeat', 'mailingFeat', 'editingFeat', 'exploreFeat',
         'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat', 'browsingFeat', 'editingFeat']])

'''

### simulate dependent data, with repeating a subseq
def Sim_depUsageData(userNum=10, subSeqLen=3, repeatPattern=None):

  df = GenUsageData(
      userNum=userNum,
      dt1=datetime.datetime(2017, 4, 11, 23, 10, 0),
      dt2=datetime.datetime(2017, 4, 12, 0, 50, 0))
  df['date'] = df['time'].dt.date
  df['date'] = df['date'].map(str)

  ## generate some pattern
  subSeq = ['photoFeat', 'locFeat', 'browsingFeat',
            'editingFeat', 'exploreFeat'][:subSeqLen]
  l = subSeqLen + 1

  if repeatPattern is None:
    repeatPattern = (len(df) // l) - 2

  Mark(repeatPattern, 'repeatPattern')
  Mark(df.shape, 'df.shape')

  for i in range(repeatPattern):
    x = subSeq + list(np.random.choice(
        a=['browsingFeat', 'mailingFeat', 'editingFeat',
           'exploreFeat', 'photoFeat',
           'PresFeat', 'locFeat'],
        size=(l-len(subSeq)),
        replace=True))

    ind = range(l*i, l*(i+1))
    booleanInd = [i in ind for i in range(len(df))]
    df.loc[booleanInd, 'prod'] = x

  return df

'''
df = Sim_depUsageData()
'''

################################ seq functions #############################
## it generates a function which for each string s
## returns a label in labelList if it is found to be a part of string s "y in s"
## but if more than one element is a match then returns MIXED
def String_isMixedFcn(labelList, mixedLabel='MIXED', noneLabel='None'):

  def F(s):

    elemInd = []

    [(y in s) for y in labelList]

    v = np.array(elemInd).sum()

    if v > 1:
      return mixedLabel
    if v == 1:
      ind = [i for i, j in enumerate(elemInd) if j]
      return labelList[ind[0]]

    return noneLabel

  return F

## adds a column to a given df about a column "col"
# including a label coming from labelList
def AddStringContainsDf(
    df,
    col,
    labelList,
    newColName=None,
    mixedLabel='MIXED',
    noneLabel='None'):

  if newColName == None:
    newColName = 'string_contains_' + col
  F = String_isMixedFcn(labelList=labelList,
                             mixedLabel=mixedLabel,
                             noneLabel=noneLabel)
  df[newColName] = df[col].map(F)

  return df

'''
labelList = ['aaa', 'ccc', 'ddd', 'bbb']
mixedLabel = 'MIXED'
noneLabel = 'None'

Fcn = String_isMixedFcn(labelList=labelList, mixedLabel='MIXED',
                             noneLabel='None')
print(Fcn('aaa-bbb-awww'))
print(Fcn('aaa-aas'))
import pandas as pd
df = pd.DataFrame({'col': ['aaa-bbb-asdsdsa', 'aaa', 'bbb', 'aaa-bbb']})

AddStringContainsDf(df=df, col='col', labelList=labelList, newColName=None,
                    mixedLabel='MIXED',
  noneLabel='None')
'''

## this adds a new column to a dataframe by inspecting a seq column
## if the seq is only unique elements, it will return the unique element
## if there are more than one elements in the seq, it returns "MIXED"
def AddSeq_uniqueOrMixed(
    df,
    seqCol,
    sepStr=None,
    newColName=None,
    mixedLabel='MIXED',
    noneLabel='None'):

  df2 = df.copy()
  df3 = df.copy()
  if sepStr is not None:
    df3 = df2.assign(**{seqCol: df2[seqCol].str.split(sepStr)})

  # for a list x, if all elements are the same, will assign the unique values,
  # otherwise will assign "mixed"
  def F(x):
    if len(x) == 0:
      return noneLabel
    if len(set(x)) == 1:
      return(x[0])
    return mixedLabel

  if newColName == None:
    newColName = seqCol + '_mix'
  df2[newColName] = df3[seqCol].map(F)

  return df2

'''
df = pd.DataFrame({
  'seq': ['a>b>c>d', 'd>e>f>t>l>h'],
  'interface': ['aa>bb>cc>dd', 'dd>ee>ff>tt>ll>hh'],
  'browser': ['aaa>aaa>aaa>aaa', 'ddd>eee>fff>ttt>lll>hhh'],
  'var2': [1, 2], 'seq_count':[5, 6]
  })

AddSeq_uniqueOrMixed(df=df, seqCol='browser', sepStr='>')
'''

## construct the set of all values for the respCol grouped by partitionCols
def GetSetIndCols(df, respCol, partitionCols):

  g = df.groupby(partitionCols)
  out = g[respCol].apply(lambda x: list(set(list(x))))
  out = out.reset_index()

  return out

'''
df = pd.DataFrame( {'a':['A', 'A', 'B', 'B', 'B', 'C', 'C'],
                    'b':[1, 2, 5, 5, 4, 6, 7]})
out = GetSetIndCols(df=df, respCol='b', partitionCols=['a'])

df = GenUsageData(8)
GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
'''

## returns a function which adds a ind column to setDf for each pair [pre, post]
def AddMembershipColFcn(setDf, setCol):

  def F(subSet):
    ind = setDf[setCol].apply(lambda x: set(subSet) < set(x))
    setDf['elems_exist'] = ind
    return (setDf)

  return F

'''
df = GenUsageData(userNum=4, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
  dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))
setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
print(setDf)
Fcn = AddMembershipColFcn(setDf=setDf, setCol='prod')
print(Fcn('k', 'j'))
print(Fcn('a', 'b'))
'''

## generates a Fcn (SubsetDfFcn) which for each given subSet:
## generates a function SubsetDf=SubsetDfFcn(subSet)
## which adds a boolean column about membership of all the elements of subSet
def ElemsExist_subsetDfFcn(setDf, setCol, partitionCols):

  AddPairMembership = AddMembershipColFcn(setDf=setDf, setCol=setCol)

  def SubsetDfFcn(subSet):
    def SubsetDf(df):
      setDf2 = AddPairMembership(subSet)
      setDf2 = setDf2[partitionCols + ['elems_exist']]
      df2 = pd.merge(df, setDf2, on=partitionCols)
      df2 = df2[df2['elems_exist']]
      return df2
    return SubsetDf

  return SubsetDfFcn


'''
df = GenUsageData(userNum=4, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
                  dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))
setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
AddPairMembership = AddMembershipColFcn(setDf=setDf, setCol='prod')
SubsetDfFcn = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod',
partitionCols=['user_id'])
SubsetDf = SubsetDfFcn(pair)

pair = ['watchFeat', 'browsingFeat']
Mark(AddPairMembership(pair))
Mark(SubsetDf(df=df)['user_id'].value_counts())
'''

## create sequence data for a variable (categorical) using timestamps and
## add blanks when there are long time time gaps
## extraCols will build a corresponding sequence for other columns of interest
## for example it can keep track of interface for a sequence of events
## we specify index columns and the sequences are built
# for each index column combination
## no deduping at this point
def CreateTimeSeq(
    df,
    respCol,
    timeCol,
    timeGap,
    timeColEnd=None,
    partitionCols=[],
    extraCols=[],
    ordered=False):

  ## we keep the main sequence column to build sequences (respCol)
  # we also keep the extra columns to build parallel sequences
  # we remove repetitions with UniqueList
  df['seq_undeduped'] = df[respCol]
  respCol = 'seq_undeduped'
  respCols = UniqueList([respCol] + extraCols)

  df2 = df[respCols].copy()
  df3 = df2[df2.isnull().any(axis=1)].copy()

  if len(df3) > 0:
    warnings.warn("\n\n the data has " + str(len(df3)) + " rows with Nulls." +
                  " CreateTimeSeq fails with Nulls." +
                  " Therefore rows with any Null are omitted."
                  " See data with Null examples below:")
    print(df3[:3])
    df2 = df2[~df2.isnull().any(axis=1)].copy()

  if not ordered:
    df = df.sort_values(partitionCols + [timeCol])

  ## creating a single column to keep track of inds
  df['all_ind'] = 0
  df['ind_change'] = False
  if len(partitionCols) > 0:
    df = Concat_stringColsDf(df, cols=partitionCols, colName='all_ind', sepStr='-')
    df['ind_pair'] = list(zip(df['all_ind'], df['all_ind'].shift()))
    df['ind_change'] = df['ind_pair'].map(lambda x: len(set(x))) > 1

  ## if durations are available by providing the end timestamp, we use that
  if timeColEnd is None:
    timeColEnd = timeCol
  df['delta'] = (df[timeCol] - df[timeColEnd].shift()).fillna(0)
  df['delta_sec'] = df['delta'].values / np.timedelta64(1, 's')
  del df['delta']

  ## we identify the beginning of new visits
  df['is_gap'] = df['delta_sec'] > timeGap
  del df['delta_sec']

  ## an ind change has the same effect as a gap
  # all we need to do is to cut the sequence at that point anyway
  # we also need to insure not to throw a way is resp is repeated
  # at this time (row)
  df['is_gap'] = df['ind_change'] | df['is_gap']
  ind = [i for i,j in enumerate(df['is_gap'].values) if j == 1]
  groupNum = len(ind) + 1
  groupsNames = range(groupNum)
  ind1 = [0] + ind + [len(df['is_gap'])]
  diff = pd.Series(ind1) - pd.Series(ind1).shift()
  diff = list(diff[1:])

  # create a column to keep track of large time gaps
  # whenever there is a gap (or ind change), we start a new group
  tempCol = []
  #Mark(zip(groupsNames, diff))
  for i in zip(groupsNames, diff):
    tempCol = tempCol + [i[0]] * int(i[1])

  df['tempCol'] = tempCol
  df['seq_start_timestamp'] = df[timeCol]
  df['seq_end_timestamp'] = df[timeColEnd]

  df = df[partitionCols + respCols +
          ['seq_start_timestamp', 'seq_end_timestamp', 'tempCol']].copy()

  #start = time.time()
  g = df.groupby(partitionCols + ['tempCol'])
  aggDict = {'seq_start_timestamp': min, 'seq_end_timestamp': max}

  for col in respCols:
    aggDict[col] = (lambda x: '>'.join(x))

  seqDf = g.agg(aggDict)
  seqDf = seqDf.reset_index()
  #end = time.time()
  #Mark(end - start, 'CreateTimeSeq running: time agg took')
  #seqDf['seq'] = seqDf[respCol]
  del seqDf['tempCol']
  #del seqDf[respCol]
  seqDf = seqDf.reset_index(drop=True)

  return seqDf

'''
# example
df = GenUsageDf_forTesting()
Mark(df[:5])

respCol = 'prod'
extraCols =['form_factor']
timeCol = 'time'
timeGap = 10*1
partitionCols = ['user_id'] # example 1
partitionCols = ['user_id', 'date'] # example 2
#df['date'] = df['date'].map(str)

timeColEnd = 'end_time'
tic = time.clock()
seqDf1 = CreateTimeSeq(
    df=df.copy(),
    respCol=respCol,
    timeCol=timeCol,
    timeGap=timeGap,
    timeColEnd=timeColEnd,
    partitionCols = partitionCols,
    extraCols=extraCols,
    ordered=False)
'''

## dedupe sequences data with parallel
def DedupeSeqDf(
  df,
  seqCol,
  extraCols=[],
  sepStr='>',
  dedupedColName='seq_deduped',
  parallelSuffix='_parallel'):

  def DedupingInd(s):
      l = s.split(sepStr)
      inds = [
          next(group)
          for key, group in itertools.groupby(
              enumerate(l), key=operator.itemgetter(1))
      ]
      ind = tuple([x[0] for x in inds])
      return ind

  def SubseqWithInd(s, ind):
    l = s.split(sepStr)
    ind = list(ind)
    out = [l[j] for j in ind]
    outString = sepStr.join(out)
    return outString

  def SubseqCol(col, subseqColName):
    df[subseqColName] = df.apply(
        lambda x: SubseqWithInd(s=x[col], ind=x['deduping_ind']), axis=1)
    return None

  df['deduping_ind'] = df[seqCol].apply(DedupingInd)

  SubseqCol(col=seqCol, subseqColName=dedupedColName)

  def ChangeColname(col):
    SubseqCol(col, subseqColName=col + parallelSuffix)

  [ChangeColname(col) for col in extraCols]

  return df

'''
df = GenUsageDf_forTesting()

respCol = 'prod'
extraCols =['form_factor', 'country']
timeCol = 'time'
timeGap = 10*1
partitionCols = ['user_id'] # example 1
partitionCols = ['user_id', 'date'] # example 2
#df['date'] = df['date'].map(str)

timeColEnd = 'end_time'
tic = time.clock()
seqDf1 = CreateTimeSeq(
    df=df.copy(),
    respCol=respCol,
    timeCol=timeCol,
    timeGap=timeGap,
    timeColEnd=timeColEnd,
    partitionCols = partitionCols,
    extraCols=extraCols,
    ordered=False)

DedupeSeqDf(df=seqDf1, seqCol='seq', extraCols=extraCols, sepStr='>',
            dedupedColName='seq_deduped', parallelSuffix='_parallel')
'''

def CreateTimeSeq_andDedupe(
    df,
    respCol,
    timeCol,
    timeGap,
    timeColEnd=None,
    partitionCols=[],
    extraCols=[],
    ordered=False,
    seqCol='seq_undeduped',
    dedupedColName='seq_deduped',
    parallelSuffix='_parallel',
    method='split_by_ind'):

  if len(set(partitionCols) & set(extraCols)) > 0:
    warnings.warn("partitionCols and extraCols intersect. This can cause errors.")

  if method == 'default':
    seqDf = CreateTimeSeq(
        df=df.copy(),
        respCol=respCol,
        timeCol=timeCol,
        timeGap=timeGap,
        timeColEnd=timeColEnd,
        partitionCols = partitionCols,
        extraCols=extraCols,
        ordered=ordered)

  elif method == 'split_by_ind':
    def CalcPerSlice(group):
      out = CreateTimeSeq(
            df=group,
            respCol=respCol,
            timeCol=timeCol,
            timeGap=timeGap,
            timeColEnd=timeColEnd,
            partitionCols = [],
            extraCols=extraCols,
            ordered=ordered)
      #Mark(out)
      for col in partitionCols:
        out[col] = group[col].values[0]
      return(out)

    if len(partitionCols) == 0:
      seqDf = CalcPerSlice(df)
    else:
      g = df.groupby(partitionCols, as_index=False)
      seqDf = g.apply(CalcPerSlice)
      seqDf = seqDf.reset_index(drop=True)
  else:
    print(
        "This method: "
        +  method
        + " is not implemented for seq calculation. we return nothing.")
    return None

  out = DedupeSeqDf(
      df=seqDf.copy(),
      seqCol=seqCol,
      extraCols=extraCols,
      sepStr='>',
      dedupedColName=dedupedColName,
      parallelSuffix=parallelSuffix)

  return out


'''
df = GenUsageDf_forTesting()

respCol = 'prod'
extraCols =['form_factor', 'country']
timeCol = 'time'
timeColEnd = 'end_time'
timeGap = 10*1
partitionCols = ['user_id'] # example 1
partitionCols = ['user_id', 'date'] # example 2
#df['date'] = df['date'].map(str)


CreateTimeSeq_andDedupe(
    df,
    respCol,
    timeCol,
    timeGap,
    timeColEnd=timeColEnd,
    partitionCols=partitionCols,
    extraCols=extraCols,
    ordered=False,
    dedupingSuffix='_deduped',
    parallelSuffix='_parallel')
'''

'''
df = GenUsageData(userNum=500,
                 dt1=datetime.datetime(2017, 4, 12),
                 dt2=datetime.datetime(2017, 4, 12, 10, 0, 0))


import cProfile
cProfile.run(
"
start = time.time()

seqDf1 = CreateTimeSeq_andDedupe(
    df=df,
    respCol='prod',
    timeCol='time',
    timeGap=1*60,
    timeColEnd='end_time',
    partitionCols=['user_id'],
    extraCols=['country'],
    ordered=True,
    dedupedColName='seq_deduped',
    parallelSuffix='_parallel',
    method='split_by_ind')
end = time.time()

Mark(end - start, 'time for split')
"
)

Mark("*******************")

cProfile.run(
"
start = time.time()

seqDf2 = CreateTimeSeq_andDedupe(
    df=df,
    respCol='prod',
    timeCol='time',
    timeGap=1*60,
    timeColEnd='end_time',
    partitionCols=['user_id'],
    extraCols=['country'],
    ordered=True,
    dedupedColName='seq_deduped',
    parallelSuffix='_parallel',
    method='default')
end = time.time()

Mark(end - start, 'time for default')
"
)
'''

## trims a seq which is given in a string format with a separator
def SeqTrim(s, k, sepStr='>'):

  ind = [pos for pos, char in enumerate(s) if char == sepStr]
  if len(ind) < k:
    return s
  else:
    return s[:ind[k - 1]]

'''
print(SeqTrim('aaa>rre>dssd>das', k=2))
print(SeqTrim('aaa>rre>dssd>das', k=1))
print(SeqTrim('aaa', k=2))
'''

## turns seq columns to basket columns
def SeqToBasketDf(
    df, cols, sepStr='>', prefix='', suffix='_basket',
    basketSepStr=';'):

  df2 = df.copy()

  def F(s):
    x = s
    if sepStr is not None:
      x = s.split(sepStr)
    out = tuple(sorted(list(set(x))))

    if basketSepStr is not None:
      out = basketSepStr.join(out)
    return out

  for col in cols:
    df2[prefix + col + suffix] = df2[col].map(F)

  return df2

'''
df = pd.DataFrame({'seq': ['a>b>c>d', 'd>e>f>t>l>h'],
  'interface': ['pc>phone>laptop>phone', 'phone>laptop>phone>phone>phone>pc'],
  'browser': ['ch>agsa>ch>agsa', 'agsa>ch>someBr>ch>someBr>ch'],
  'var2': [1, 2], 'seq_count':[5, 6]})
print(df)

SeqToBasketDf(df=df, cols=['seq', 'interface', 'browser'],
              sepStr='>', prefix='', suffix='_basket')
'''

## complete deduping of a sequence,
# we only keep the first occurrence of an elem in the seq
def Seq_completeDedupeDf(
  df, cols, sepStr='>', prefix='', suffix='_completely_deduped'):

  df2 = df.copy()

  def F(s):
    x = s
    if sepStr is not None:
      x = s.split(sepStr)
    out = UniqueList(x)
    if sepStr is not None:
      out = sepStr.join(out)
    return out

  for col in cols:
    df2[prefix + col + suffix] = df2[col].map(F)

  return df2

'''
df = pd.DataFrame({
    'seq': ['a>a>c>d>a', 'd>e>f>d>l>h'],
    'interface': ['pc>phone>laptop>phone', 'phone>laptop>phone>phone>phone>pc'],
    'browser': ['ch>agsa>ch>agsa', 'agsa>ch>someBr>ch>someBr>ch'],
    'var2': [1, 2],
    'seq_count': [5, 6]})

print(df)

Seq_completeDedupeDf(
    df=df,
    cols=['seq', 'interface', 'browser'],
    sepStr='>',
    prefix='',
    suffix='_completely_deduped')
'''

### shift augmenting
## for a given sequence given in string format it constructs sequences
# by shifting the given seq one by one
# it insures the resulting sequences are of length k
# unless the full seq is shorter
# it also provides the lag sequence
def SeqShiftedList(
    s, k=None, lagK=None, sepStr=None, basketSepStr=',', blankStr='BLANK'):

  if k == None:
    k = len(s)
  seq = s
  if sepStr != None:
    seq = s.split(sepStr)
  l = len(seq)
  if l <= k:
    return ({'lagSeq':[blankStr], 'lagBasket':[blankStr], 'shifted':[s]})

  shifted = []
  lagSeq = []
  lagBasket = []

  for i in range(l-k+1):
    seq0 = seq[i:(i+k)]
    m = 0
    if lagK != None:
      m = max([i - lagK, 0])
    lag0 = seq[m:i]
    basket0 = tuple(sorted(list(set(lag0))))
    if basketSepStr != None:
      basket0 = ','.join(basket0)

    if sepStr != None:
      seq0 = sepStr.join(seq0)
      lag0 = sepStr.join(lag0)

    ## resetting empty string with BLANK
    if lag0 == '':
      lag0 = blankStr
      basket0 = blankStr

    shifted.append(seq0)
    lagSeq.append(lag0)
    lagBasket.append(basket0)
  outDict = {'lagSeq':lagSeq, 'lagBasket':lagBasket, 'shifted':shifted}

  return outDict

'''
s = 'a>b>c>d'
SeqShiftedList(s=s, k=2, sepStr=None)
'''

## creates a data frame by shifting sequences
# and creating multiple sequences from one
# if extraCols are also presented the same will be done to extraCols
def ShiftedSeqDf(
    df,
    seqCol,
    k=None,
    lagK=None,
    sepStr='>',
    extraCols=[],
    colPrefix='trimmed_',
    colSuffix=''):

  ## add shifted
  def F(s):
    return SeqShiftedList(s=s, k=k, sepStr=sepStr)['shifted']

  ## add start time (omitted, this would work only if we included all shifts,
  #but now we are not including short shifts)
  #def G(s):
  #  if sepStr == None:
  #    return(range(len(s)))
  #  return(range(len(s.split(sepStr))))

  ## add lagSeq
  def H(s):
    return SeqShiftedList(s=s, k=k, lagK=lagK, sepStr=sepStr)['lagSeq']

  def B(s):
    return SeqShiftedList(s=s, k=k, lagK=lagK, sepStr=sepStr)['lagBasket']

  df2 = df.copy()
  df2[seqCol] = [F(x) for x in df[seqCol].values]
  df2['seq_shift_order'] = [range(len(x)) for x in df2[seqCol].values]

  seqDf = Flatten_RepField(df=df2, listCol=seqCol, sep=None)

  df2['lag'] = [H(x) for x in df[seqCol].values]
  seqDf2 = Flatten_RepField(df=df2, listCol='lag', sep=None)
  seqDf['lag'] = seqDf2['lag']

  df2['lagBasket'] = [B(x) for x in df[seqCol].values]
  seqDf2 = Flatten_RepField(df=df2, listCol='lagBasket', sep=None)
  seqDf['lagBasket'] = seqDf2['lagBasket']

  for col in (extraCols):
    df2[col] = [F(x) for x in df2[col].values]
    seqDf2 = Flatten_RepField(df=df2, listCol=col, sep=None)
    seqDf[col] = seqDf2[col]

    df2[col + '_lag'] = [H(x) for x in df[col].values]
    seqDf2 = Flatten_RepField(df=df2, listCol=col + '_lag', sep=None)
    seqDf[col + '_lag'] = seqDf2[col + '_lag']

    df2[col + '_lagBasket'] = [B(x) for x in df[col].values]
    seqDf2 = Flatten_RepField(df=df2, listCol=col + '_lagBasket', sep=None)
    seqDf[col + '_lagBasket'] = seqDf2[col + '_lagBasket']

  orderDf = Flatten_RepField(df=df2, listCol='seq_shift_order', sep=None)
  seqDf['seq_shift_order'] = orderDf['seq_shift_order'].map(str)

  for col in ([seqCol] + extraCols):
    seqDf.rename(columns={col: colPrefix + col + colSuffix}, inplace=True)

  return seqDf

'''
df = pd.DataFrame({'seq': ['a>b>c>d', 'd>e>f>t>l>h'],
  'interface': ['pc>phone>laptop>phone', 'phone>laptop>phone>phone>phone>pc'],
  'browser': ['ch>agsa>ch>agsa', 'agsa>ch>someBr>ch>someBr>ch'],
  'var2': [1, 2], 'seq_count':[5, 6]})
print(df)
ShiftedSeqDf(df=df, seqCol='seq', k=3, lagK=2, sepStr='>',
             extraCols=['interface', 'browser'])
'''

## adds condition columns to seq data, eg a condition for the 2nd seq element
# this is useful to study the entry points to an app,
# because we can condition the second element of the seq to be equal
# to that app of interest
def AddSeqOrdEvent(
    df, seqCol, sepStr=None, basketSepStr=',', noneValue='BLANK'):

  df2 = df.copy()
  if sepStr != None:
    df3 = df2.assign(**{seqCol: df2[seqCol].str.split(sepStr)})
  def GetListKthElFcn(k, noneValue=None):

    def F(x):
      if len(x) <= k:
        return noneValue
      else:
        return x[k]

    return F

  F1st = GetListKthElFcn(k=0, noneValue=noneValue)
  F2nd = GetListKthElFcn(k=1, noneValue=noneValue)
  F3rd = GetListKthElFcn(k=2, noneValue=noneValue)
  F4th = GetListKthElFcn(k=3, noneValue=noneValue)
  df2['event_1'] =[F1st(x) for x in df3[seqCol].values]
  df2['event_2'] = [F2nd(x) for x in df3[seqCol].values]
  df2['event_3'] = [F3rd(x) for x in df3[seqCol].values]
  df2['event_4'] = [F4th(x) for x in df3[seqCol].values]

  ## add subseq info
  if sepStr is not None:
    df2['subseq_1_2'] = df2['event_1'] + sepStr + df2['event_2']
    df2['subseq_1_2_3'] = df2['subseq_1_2'] + sepStr + df2['event_3']
    df2['subseq_1_2_3_4'] = df2['subseq_1_2_3'] + sepStr + df2['event_4']
    for col in ['subseq_1_2', 'subseq_1_2_3', 'subseq_1_2_3_4']:
      df2[col] = df2[col].map(lambda x: x.replace((sepStr + noneValue), ''))
  else:
    df2['subseq_1_2'] = df2[['event_1', 'event_2']].apply(
        lambda x: tuple(x), axis=1)
    df2['subseq_1_2_3'] = df2[['event_1', 'event_2', 'event_3']].apply(
        lambda x: tuple(x), axis=1)
    df2['subseq_1_2_3_4'] = df2[['event_1', 'event_2', 'event_3', 'event_4']].apply(
        lambda x: tuple(x), axis=1)

  ## add basketed data (set of events, rather than subseq)
  df2['basket_1_2'] = df2[['event_1', 'event_2']].apply(
        lambda x: basketSepStr.join(sorted(list(set(x)))), axis=1)
  df2['basket_1_2_3'] = df2[['event_1', 'event_2', 'event_3']].apply(
        lambda x: basketSepStr.join(sorted(list(set(x)))), axis=1)
  df2['basket_1_2_3_4'] = df2[['event_1', 'event_2', 'event_3', 'event_4']].apply(
        lambda x: basketSepStr.join(sorted(list(set(x)))), axis=1)

  return df2


'''
df = pd.DataFrame({'seq': ['a>b>c>d', 'd>e>f>t>l>h'], 'var2': [1, 2],
                   'seq_count':[5, 6]})
print(df)
seqDf = ShiftedSeqDf(df=df, seqCol='seq', k=3, sepStr='>')
AddSeqOrdEvent(df=seqDf, seqCol='seq', sepStr='>')
'''

'''
size = 8
df = pd.DataFrame({
    'seq':np.random.choice(
        a=['a>b>c>d', 'b>c>f>g', 'f>c>v>f>g>a>b'],
        size=8,
        replace=True),
    'col1':np.random.uniform(low=0.0, high=100.0, size=size),
    'col2':np.random.uniform(low=0.0, high=100.0, size=size),
    'col3':np.random.uniform(low=0.0, high=100.0, size=size),
    'col4':np.random.uniform(low=0.0, high=100.0, size=size)})

#print(df)
seqDf = ShiftedSeqDf(df=df, seqCol='seq', k=3, sepStr='>')
AddSeqOrdEvent(df=seqDf, seqCol='seq')
'''
'''
df0 = pd.DataFrame({
    'categ':np.random.choice(
        a=['a>b>c>d', 'b>c>f>g', 'f>c>v>f>g>a>b'],
        size=8,
        replace=True),
    'col1':np.random.uniform(low=0.0, high=100.0, size=5),
    'col2':np.random.uniform(low=0.0, high=100.0, size=5),
    'col3':np.random.uniform(low=0.0, high=100.0, size=5),
    'col4':np.random.uniform(low=0.0, high=100.0, size=5)})

print(df)
seqDf = ShiftedSeqDf(df=df, seqCol=seqCol, k=3, sepStr='>')
AddSeqOrdEvent(df=seqDf, seqCol='seq')
'''

## adds a sequence length column to seq data
def AddSeqLength(df, seqCol, seqLenCol='seq_length', sepStr=None):

  df2 = df.copy()
  df3 = df2.copy()

  if sepStr is not None:
    df3 = df2.assign(**{seqCol: df2[seqCol].str.split(sepStr)})

  def F(x):
    return len(x)

  df2[seqLenCol] = [F(x) for x in df3[seqCol].values]

  return df2

'''
df = pd.DataFrame({
    'seq': ['a>b>c>d', 'd>e>f>t>l>h'],
    'var2': [1, 2],
    'seq_count':[5, 6]})
print(df)
seqDf = ShiftedSeqDf(df=df, seqCol='seq', k=3, sepStr='>')
seqDf2 = AddSeqOrdEvent(df=seqDf, seqCol='seq', sepStr='>')
seqDf3 = AddSeqLength(df=seqDf, seqCol='seq', sepStr='>')
'''

## dedupe consecutive elements repetitions in a seq
def DedupeSeq(s, sepStr=None):

  if sepStr != None:
    s = s.split(sepStr)
  out = [x[0] for x in itertools.groupby(s)]

  if sepStr != None:
    out = sepStr.join(out)

  return(out)

'''
DedupeSeq(s='a>a>b>b>a>a', sepStr='>')
DedupeSeq(s='a', sepStr='>')
DedupeSeq(s='ab', sepStr='>')
DedupeSeq(s=['a'], sepStr=None)
DedupeSeq(s=['a', 'a', 'b', 'b'], sepStr=None)
'''

## creates a seq table
# which also includes slicing based on what boolean columns
# for the seq containing prods
# seqDimCols are the dimensions used in the sequence definition,
# for example if seqDimCols=['prod', 'form_factor']
# then the sequence elements look like: mailingFeat-COMP
# partitionCols are the dimensions for which we slice the sequence data
# keepIndCols specifies if we should also keep the partitionCols in the seq table
# e.g. partitionCols=['user_id', 'date'] would insure
# that the sequences for each [user and date] are separated
#(in different rows)
# this only happens if keepIndCols = True
# we always shift but keep track of event order
# condDict is for slicing the data
def CreateSeqDf(
    df,
    timeCol,
    seqDimCols,
    partitionCols,
    timeGap,
    trim,
    keepTimeCols=False,
    timeColEnd=None,
    extraCols=[],
    extraColsDeduped=[],
    seqIdCols=None,
    addOrigSeqInfo=True,
    addBasket=True,
    addLagInfo=True,
    lagTrim=2,
    ordered=False,
    method='split_by_ind',
    addResetDate_seqStartDate=True):

  df = df.reset_index(drop=True)

  #Mark(df)
  df = Concat_stringColsDf(df, cols=seqDimCols, colName=None, sepStr='-')
  respCol = '-'.join(seqDimCols)

  seqDf = CreateTimeSeq_andDedupe(
      df=df,
      respCol=respCol,
      timeCol=timeCol,
      timeGap=timeGap,
      timeColEnd=timeColEnd,
      partitionCols=partitionCols,
      extraCols=extraCols,
      ordered=ordered,
      seqCol='seq_undeduped',
      dedupedColName='seq_deduped',
      parallelSuffix='_parallel',
      method=method)

  extraCols_parallel = [s + '_parallel' for s in extraCols]

  ## reset an existing (or add) date column to be seq_start_date
  if addResetDate_seqStartDate:
    seqDf['date'] = seqDf['seq_start_timestamp'].dt.date

  seqDf['full_seq_dur_secs'] =  (seqDf['seq_end_timestamp'] -
    seqDf['seq_start_timestamp']).values / np.timedelta64(1, 's')

  for col in ['seq_start_timestamp', 'seq_end_timestamp', 'date']:
    if col in seqDf.columns:
      seqDf[col] = seqDf[col].map(str)

  ## assign seq id
  if seqIdCols is None:
    seqIdCols = partitionCols

  seqDf = Concat_stringColsDf(
      df=seqDf, cols=seqIdCols, colName='seq_id', sepStr='-')
  seqDf['seq_id'] =  seqDf['seq_id'] + '-' + seqDf['seq_start_timestamp']

  ## add basket info for full sequences (not trimmed)
  if addOrigSeqInfo:
    seqDf = SeqToBasketDf(
        df=seqDf,
        cols=(['seq_deduped'] + extraCols),
        sepStr='>',
        prefix='full_',
        suffix='_basket')

    ## add completely deduped versions for the full sequences
    seqDf = Seq_completeDedupeDf(
        df=seqDf,
        cols=(['seq_deduped'] + extraCols),
        sepStr='>',
        prefix='full_',
        suffix='_completely_deduped')

    for col in (['seq_undeduped', 'seq_deduped'] +
                extraCols + extraCols_parallel):
      seqDf['full_' + col] = seqDf[col]

    ## fix the over expressive column names
    seqDf.rename(
        columns={
            'full_seq_deduped_completely_deduped': 'full_seq_completely_deduped',
            'full_seq_deduped_basket': 'full_seq_basket'},
        inplace=True)

  seqDf = AddSeqLength(
      df=seqDf,
      seqCol='seq_undeduped',
      seqLenCol='full_seq_undeduped_length',
      sepStr='>')

  seqDf = AddSeqLength(
      df=seqDf,
      seqCol='seq_deduped',
      seqLenCol='full_seq_deduped_length',
      sepStr='>')

  #Mark(seqDf[:5])
  ## shift-augmenting the sequences
  # this will also add the seq_shift_order
  # in case we want to get back un-shifted seq only
  seqDf = ShiftedSeqDf(
      df=seqDf,
      seqCol='seq_deduped',
      k=trim,
      lagK=lagTrim,
      sepStr='>',
      extraCols=extraCols_parallel)

  seqDf['shifted_seq_id'] =  seqDf['seq_id'] + '-' + seqDf['seq_shift_order']

  ## adding event order
  seqDf = AddSeqOrdEvent(
      df=seqDf.copy(),
      seqCol='trimmed_seq_deduped',
      sepStr='>',
      noneValue='BLANK')

  ## adding baskets to the shifted sequences and their extra columns
  seqDf = SeqToBasketDf(
    df=seqDf,
    cols=(
        ['trimmed_seq_deduped']
        + [('trimmed_'+ x) for x in extraCols_parallel]),
    sepStr='>',
    prefix='',
    suffix='_basket')

  ## changing overly expressive column names
  # (basket is more reduction that deduping so we drop deduped)
  seqDf.rename(
        columns={
            'trimmed_seq_deduped_basket': 'trimmed_sequence_basket'},
        inplace=True)

  seqDf['trimmed_seq_count'] = 1

  ## adding seq length
  seqDf = AddSeqLength(
      df=seqDf,
      seqCol='trimmed_seq_deduped',
      seqLenCol='trimmed_seq_deduped_length',
      sepStr='>')

  ## for each extraCol (e.g. interface)
  # we check if the corresponding seq is mixed or same
  for col in extraCols:
    seqDf = AddSeq_uniqueOrMixed(
        df=seqDf,
        seqCol='trimmed_' + col + '_parallel',
        newColName='trimmed_' + col + '_parallel' + '_mix',
        sepStr='>')

  for col in extraColsDeduped:
    seqDf['trimmed_' + col + '_seq_deduped'] = (
        seqDf['trimmed_' + col + '_parallel'].map(
            lambda s: DedupeSeq(s, sepStr='>')))

  ## removing the columns created on the fly which are not needed.
  for col in ['seq_undeduped'] + extraCols:
    del seqDf[col]

  return seqDf


'''
df = GenUsageDf_forTesting()

extraCols =['form_factor']
timeCol = 'time'
timeGap = 10*1
timeColEnd = 'end_time'
trim = 3

## Example
seqDf = CreateSeqDf(
    df=df,
    timeCol='time',
    seqDimCols=['prod', 'form_factor'],
    partitionCols=['user_id'],
    timeGap=timeGap,
    trim=trim,
    keepTimeCols=True,
    timeColEnd=timeColEnd,
    extraCols=extraCols,
    ordered=True)

for col in list(seqDf.columns):
  print(col)

seqDf[['full_seq_undeduped',
       'full_seq_deduped',
       'trimmed_seq_deduped',
       'trimmed_seq_deduped_basket',
       'trimmed_form_factor_parallel',
       'trimmed_form_factor_parallel_basket',
       'trimmed_form_factor_parallel_mix']]

'''


####
'''
df = GenUsageData(userNum=50, dt1=datetime.datetime(2017, 4, 12, 23, 0, 0),
             dt2=datetime.datetime(2017, 4, 13, 2, 0, 0))

## Example 1
seqDf1 = CreateSeqDf(
    df=df,
    timeCol='time',
    seqDimCols=['prod', 'form_factor'],
    partitionCols=['user_id'],
    timeGap=1*60,
    trim=3,
    extraCols=[],
    ordered=True)

Mark(seqDf1)


## Example 2
df = GenUsageData(userNum=50, dt1=datetime.datetime(2017, 4, 12, 23, 0, 0),
             dt2=datetime.datetime(2017, 4, 13, 2, 0, 0))

seqDf2 = CreateSeqDf(
    df=df,
    timeCol='time',
    seqDimCols=['prod', 'form_factor'],
    partitionCols=['user_id'],
    timeGap=1*60,
    trim=3,
    extraCols=['prod', 'form_factor'],
    ordered=True)
'''

'''
## better example
df = pd.DataFrame(columns=['country', 'user_id', 'date', 'time', 'end_time', 'prod', 'form_factor'])
df.loc[0] =       ['US', '0', '2017-04-12', '2017-04-12 00:03:00', '2017-04-12 00:04:00', 'PresFeat', 'COMP']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:04:01', '2017-04-12 00:05:03', 'photoFeat', 'COMP']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:05:05', '2017-04-12 00:06:04', 'PresFeat', 'PHN']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:06:05', '2017-04-12 00:06:08', 'PresFeat', 'PHN']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:06:30', '2017-04-12 00:06:45', 'exploreFeat', 'COMP']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:07:00', '2017-04-12 00:07:50', 'editingFeat', 'PHN']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:14:00', '2017-04-12 00:14:10', 'photoFeat', 'COMP']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:16:00', '2017-04-12 00:18:59', 'locFeat', 'COMP']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:19:00', '2017-04-12 00:20:00', 'locFeat', 'COMP']
df.loc[len(df)] = ['US', '0', '2017-04-12', '2017-04-12 00:22:00', '2017-04-12 00:22:00', 'browsingFeat', 'PHN']

for col in ['time', 'end_time']:
  df[col] = df[col].map(ConvertToDateTimeFcn())

extraCols =['form_factor']
timeCol = 'time'
timeGap = 10*1
timeColEnd = 'end_time'
trim = 3

## Example
seqDf = CreateSeqDf(
    df=df,
    timeCol='time',
    seqDimCols=['prod', 'form_factor'],
    partitionCols=['user_id'],
    timeGap=timeGap,
    trim=trim,
    keepTimeCols=True,
    timeColEnd=timeColEnd,
    extraCols=[],
    ordered=True)

'''

## this function builds sequential data and
def BuildAndWriteSeqDf(
    df,
    fn,
    seqDimCols,
    partitionCols,
    timeGap,
    trim,
    timeCol,
    keepTimeCols=False,
    timeColEnd=None,
    seqPropCols=[],
    seqPropColsDeduped=[],
    seqIdCols=None,
    writePath='',
    addOrigSeqInfo=True,
    addBasket=True,
    addLagInfo=False,
    lagTrim=3,
    ordered=False,
    method='split_by_ind',
    addResetDate_seqStartDate=True):

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

  seqDf = CreateSeqDf(
      df=df,
      timeCol=timeCol,
      seqDimCols=seqDimCols,
      partitionCols=partitionCols,
      timeGap=timeGap,
      trim=trim,
      keepTimeCols=keepTimeCols,
      timeColEnd=timeColEnd,
      extraCols=seqPropCols,
      extraColsDeduped=seqPropColsDeduped,
      seqIdCols=seqIdCols,
      addOrigSeqInfo=addOrigSeqInfo,
      addBasket=addBasket,
      addLagInfo=addLagInfo,
      lagTrim=lagTrim,
      ordered=ordered,
      method=method,
      addResetDate_seqStartDate=addResetDate_seqStartDate)

  '''
  for col in ['trimmed_seq_deduped_length',
              'full_seq_deduped_length',
              'full_seq_undeduped_length']:
    if col in list(seqDf.columns):
      seqDf[col] = seqDf[col].map(str)
  '''
  #Mark(seqDf)
  del seqDf['deduping_ind']

  if fn != None:
    WriteCsv(df=seqDf, fn=writePath + fn + '.csv', printLog=True)

  return seqDf

## create random sequences
def GenRandomSeq(size, pvals=[0.5, 0.1] + [0.05]*8):

  x = np.random.multinomial(n=10, pvals=pvals, size=size)
  df = pd.DataFrame({'seq': x.tolist()})
  def F(x):
    x = [str(y) for y in x]
    return '>'.join(x)
  df['seq'] = df['seq'].map(F)
  df['seq_count'] = np.random.poisson(lam=5.0, size=size)

  return df

# simulation for sig odds
def SimulationSeqOdds():

  df = GenRandomSeq(size=1000, pvals=[0.5, 0.3, 0.1, 0.1])

  res = SeqConfIntDf(
      df=df,
      seqCol='seq',
      Fcn=SeqTransOddsFcn,
      seqCountCol='seq_count',
      shift=True,
      sepStr='>',
      bsSize=200)

  res['odds'].map(np.log).hist()
  plt.title('log odds')

  return res

'''
res = SimulationSeqOdds()
print(res['lower'] > 2).mean()
print(res['upper'] < 0.5).mean()
'''
