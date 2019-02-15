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

"""Function for working with data in python
it includes some all purpose functions to work with dataframes
it includes some plotting functions
"""

###############################  PART 0: Generic all purpose python functions
## this prints an object and mark it
## good for debugging
def Mark(x='', text='', color='', bold=True,
         underline=False):
  start = ''
  end = ''
  if color != '' or bold or underline:
    end='\033[0m'
  colorDict = {
      '': '',
      None: '',
      'purple' : '\033[95m',
      'cyan' : '\033[96m',
      'darkcyan' : '\033[36m',
      'blue' : '\033[94m',
      'green' : '\033[92m',
      'yellow' : '\033[93m',
      'red' : '\033[91m',
      'bold' : '\033[1m',
      'underline' : '\033[4m',
      'end' : '\033[0m'}
  if bold:
    start = start + colorDict['bold']
  if underline:
    start = start + colorDict['underline']
  start = start + colorDict[color]

  print("*** " + start + text + end)
  if x is not None:
    print(x)

'''
## examples
Mark(2*10, 'xxx', bold=False, underline=True)
Mark(2*10, 'xxx', bold=False, underline=True, color='red')
Mark(x='', text='xxx', bold=True, underline=True, color='green')
'''

# to print a function definition
PrintFcn = inspect.getsourcelines

## mapping a dictionary via map
def MapDict(f, dic):
  return dict(map(lambda (k,v): (k, f(v)), dic.iteritems()))

def BitOr(x):
  return functools.reduce(lambda a,b: (a|b), x)

## builds a function for rounding up to n number of significant digits
def Signif(n):

  def F(x):
    if math.isnan(x):
      return(x)
    if x == 0:
      return(0)
    out = round(np.absolute(x),
                -int(math.floor(math.log10(np.absolute(x))) + (-n+1)))
    if x < 0:
      out = -out
    return(out)

  return F


###Part 1: Data frame and data wrangling functions

## generate a data frame manually for testing data frame functions and usage metrics
def GenUsageDf_forTesting():

  df = pd.DataFrame(columns=['country', 'user_id', 'expt', 'date', 'time', 'end_time', 'prod', 'form_factor'])
  df.loc[len(df)] = ['US', '0', 'base', '2017-04-12', '2017-04-12 00:03:00', '2017-04-12 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['US', '0', 'base', '2017-04-12', '2017-04-12 00:04:01', '2017-04-12 00:05:03', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['US', '0', 'base', '2017-04-12', '2017-04-12 00:05:05', '2017-04-12 00:06:04', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '0', 'base', '2017-04-12', '2017-04-12 00:06:05', '2017-04-12 00:06:08', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '0', 'base', '2017-04-12', '2017-04-12 00:06:30', '2017-04-12 00:06:45', 'SHEETS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:07:00', '2017-04-12 00:07:50', 'DOCS', 'PHONE']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:14:00', '2017-04-12 00:14:10', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:16:00', '2017-04-12 00:17:09', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base',  '2017-04-12', '2017-04-12 00:18:00', '2017-04-12 00:18:30', 'randomBrowseApp', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:20:00', '2017-04-12 00:21:00', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:22:00', '2017-04-12 00:22:00', 'randomBrowseApp', 'PHONE']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:03:00', '2017-04-12 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:04:01', '2017-04-12 00:05:03', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '1', 'base', '2017-04-12', '2017-04-12 00:05:05', '2017-04-12 00:06:04', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '2', 'base', '2017-04-12', '2017-04-12 00:06:05', '2017-04-12 00:06:08', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '2', 'base', '2017-04-12', '2017-04-12 00:06:30', '2017-04-12 00:06:45', 'SHEETS', 'COMPUTER']
  df.loc[len(df)] = ['US', '2', 'base', '2017-04-12', '2017-04-12 00:07:00', '2017-04-12 00:07:50', 'DOCS', 'PHONE']
  df.loc[len(df)] = ['JP', '3', 'test', '2017-04-12', '2017-04-12 00:14:00', '2017-04-12 00:14:10', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '3', 'test', '2017-04-12', '2017-04-12 00:14:20', '2017-04-12 00:18:59', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '3', 'test', '2017-04-12', '2017-04-12 00:19:00', '2017-04-12 00:20:00', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '3', 'test', '2017-04-12', '2017-04-12 00:20:20', '2017-04-12 00:22:00', 'randomBrowseApp', 'PHONE']
  df.loc[len(df)] = ['US', '4', 'test', '2017-04-14', '2017-04-14 00:03:10', '2017-04-14 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['US', '4', 'test', '2017-04-14', '2017-04-14 00:04:10', '2017-04-14 00:05:03', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['US', '4', 'test', '2017-04-14', '2017-04-14 00:05:15', '2017-04-14 00:06:04', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '4', 'test', '2017-04-14', '2017-04-14 00:06:01', '2017-04-14 00:06:08', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '4', 'test', '2017-04-14', '2017-04-14 00:06:35', '2017-04-14 00:06:45', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['US', '5', 'test', '2017-04-14', '2017-04-14 00:03:07', '2017-04-14 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['US', '5', 'test', '2017-04-14', '2017-04-14 00:04:04', '2017-04-14 00:05:03', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['US', '5', 'test', '2017-04-14', '2017-04-14 00:05:04', '2017-04-14 00:06:04', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '5', 'test', '2017-04-14', '2017-04-14 00:06:03', '2017-04-14 00:06:08', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['US', '5', 'test', '2017-04-14', '2017-04-14 00:06:28', '2017-04-14 00:06:45', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['JP', '6', 'test', '2017-04-14', '2017-04-14 00:14:01', '2017-04-14 00:14:10', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '6', 'test', '2017-04-14', '2017-04-14 00:14:19', '2017-04-14 00:18:59', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '6', 'test', '2017-04-14', '2017-04-14 00:19:10', '2017-04-14 00:20:00', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '6', 'test', '2017-04-14', '2017-04-14 00:20:11', '2017-04-14 00:22:00', 'randomBrowseApp', 'PHONE']
  df.loc[len(df)] = ['JP', '7', 'base', '2017-04-15', '2017-04-15 00:14:11', '2017-04-15 00:14:10', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '7', 'base', '2017-04-15', '2017-04-15 00:14:22', '2017-04-15 00:18:59', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '7', 'base', '2017-04-15', '2017-04-15 00:19:57', '2017-04-15 00:20:00', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['JP', '7', 'base', '2017-04-15', '2017-04-15 00:21:56', '2017-04-15 00:22:00', 'randomBrowseApp', 'PHONE']
  df.loc[len(df)] = ['FR', '8', 'base', '2017-04-12', '2017-04-12 00:03:00', '2017-04-12 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['FR', '8', 'base', '2017-04-12', '2017-04-12 00:04:01', '2017-04-12 00:05:03', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['FR', '8', 'base', '2017-04-12', '2017-04-12 00:05:05', '2017-04-12 00:06:04', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['FR', '8', 'base', '2017-04-12', '2017-04-12 00:06:05', '2017-04-12 00:06:08', 'SLIDES', 'PHONE']
  df.loc[len(df)] = ['FR', '8', 'base', '2017-04-12', '2017-04-12 00:06:30', '2017-04-12 00:06:45', 'SHEETS', 'COMPUTER']
  df.loc[len(df)] = ['FR', '9', 'test', '2017-04-15', '2017-04-15 00:14:11', '2017-04-15 00:14:10', 'PHOTOS', 'COMPUTER']
  df.loc[len(df)] = ['FR', '9', 'test', '2017-04-15', '2017-04-15 00:14:22', '2017-04-15 00:18:59', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['FR', '9', 'test', '2017-04-15', '2017-04-15 00:19:57', '2017-04-15 00:20:00', 'MAPS', 'COMPUTER']
  df.loc[len(df)] = ['FR', '9', 'test', '2017-04-15', '2017-04-15 00:21:56', '2017-04-15 00:22:00', 'randomBrowseApp', 'PHONE']
  df.loc[len(df)] = ['NG', '10', 'test', '2017-04-16', '2017-04-15 00:21:56', '2017-04-15 00:22:00', 'DRIVE', 'PHONE']
  df.loc[len(df)] = ['IR', '11', 'test', '2017-04-12', '2017-04-15 00:21:56', '2017-04-15 00:22:00', 'randomBrowseApp', 'PHONE']
  df.loc[len(df)] = ['IR', '12', 'base', '2017-04-16', '2017-04-15 00:21:56', '2017-04-15 00:22:00', 'randomWatchApp', 'PHONE']
  df.loc[len(df)] = ['IR', '13', 'base', '2017-04-12', '2017-04-12 00:03:00', '2017-04-12 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['RU', '14', 'base', '2017-04-12', '2017-04-12 00:03:00', '2017-04-12 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['RU', '15', 'base', '2017-04-13', '2017-04-13 00:03:00', '2017-04-13 00:04:00', 'SLIDES', 'COMPUTER']
  df.loc[len(df)] = ['RU', '16', 'base', '2017-04-14', '2017-04-14 00:03:00', '2017-04-14 00:04:00', 'SLIDES', 'COMPUTER']

  def F(x):
    return(datetime.datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))

  for col in ['time', 'end_time']:
    df[col] = df[col].map(F)

  df['duration'] = (df['end_time'] - df['time']) / np.timedelta64(1, 's')
  df['value'] = np.random.uniform(low=1.0, high=5.0, size=df.shape[0])

  return df

## this subsets a df according to values given in the dict: condDict
# the data columns are given in the dictionary keys
# the possible values (a list of values) for each column are
# given in the dict values
def BuildCondInd(df, condDict):

  cols = condDict.keys()
  n = df.shape[0]
  ind = pd.Series([True] * n)
  for i in range(len(cols)):
    col = cols[i]
    valueList = condDict[col]
    if valueList != None and valueList != []:
      ind0 = (df[col].isin(valueList))
      ind = ind * ind0

  return ind

'''
df = pd.DataFrame({'a':[2, 1, 3, 2, 2, 2], 'b':['A', 'A', 'B', 'C', 'C', 'C'],
  'c':['11','22','22','22', '22', '22']})
ind = BuildCondInd(df=df, condDict={'a':[1, 2], 'b':['A', 'B']})
df[ind]

ind = BuildCondInd(df=df, condDict={'a':[1, 2], 'b':None})
df[ind]
'''

## get the sub df immediately
def SubDf_withCond(df, condDict, resetIndex=True):

  if (condDict is None) or (len(condDict) == 0):
    return df

  df = df.reset_index(drop=True)
  ind = BuildCondInd(df=df, condDict=condDict)
  df2 = df[ind].copy()

  if resetIndex:
    df2 = df2.reset_index(drop=True)

  return df2


## subset df based on regex filters on string columns
# every column is given in a key and the value is a regex
def BuildRegexInd(df, regDict):

  cols = regDict.keys()
  n = df.shape[0]
  ind = pd.Series([True]*n)
  for i in range(len(cols)):
    col = cols[i]
    valueList = regDict[col]
    if valueList != None and valueList != []:
      ind0 = pd.Series([False] * n)
      for value in valueList:
        ind0 = ind0 + df[col].map(str).str.contains(value)

    ind = ind * ind0
    return ind

'''
df = pd.DataFrame(
    {'a':[24, 12, 63, 2, 3312, 2],
     'b':['A', 'A', 'BBAA', 'CD', 'CE', 'CF'],
     'c':['11','22','22','23', '22', '22']})

ind = BuildRegexInd(df=df, regDict={'a':['1', '2'], 'b':['A', 'B']})
Mark(df[ind])

ind = BuildRegexInd(df=df, regDict={'a':['1', '3'], 'b':None})
Mark(df[ind])

ind = BuildRegexInd(df=df, regDict={'b':['B', 'C'], 'b':['^(?:(?!CE).)*$']})
Mark(df[ind])

## column b does not include CE but it includes A or B.
ind = BuildRegexInd(df=df, regDict={'b':['^(?!.*CE).*B.*$', '^(?!.*CE).*A.*$']})
Mark(df[ind])
'''

## check for two strings regex
def IncludeBothStrRegex(s1, s2):

  out = '^(?=.*' + s1 + ')(?=.*' + s2 + ').*$'
  return out

'''
reg = IncludeBothStrRegex(' cat ', ' dog ')
print(reg)
print(pd.Series(['cat-dog', ' cat hates dog ', 'tiger']).str.contains(reg))
'''

## rehashing the sensitive a column (col)
## the input is a dictionary of data frames with that column
## we make sure the rehashing is fixed across data frames
def RehashColDfDict(dfDict, col, newCol=None, omitCol=False):

  if newCol == None:
    newCol = col + '_hashed'
  dfNames = dfDict.keys()
  values = []
  for key in dfNames:
    df0 = dfDict[key]
    values0 = df0[col].values
    values = list(set(values + list(values0)))
  dfHash = pd.DataFrame({col: values, 'tempCol': range(len(values))})
  newDfDict = {}
  for key in dfNames:
    df0 = dfDict[key]
    dfNew = pd.merge(df0, dfHash, on=[col], how='left')
    dfNew[newCol] = dfNew['tempCol']
    del dfNew['tempCol']
    if omitCol:
      del dfNew[col]
    newDfDict[key] = dfNew

  return newDfDict

# it converts a float or string date to datetime
def FloatOrStrToDate(x, format="%Y%m%d"):

  if (x == None) or (x == 'nan') or (x == np.nan):
    return pd.NaT
  if (type(x).__name__ == 'float') and math.isnan(x):
    return pd.NaT
  s = str(x)
  if (s == 'nan'):
    return pd.NaT
  import re
  s = re.sub('_', '', s)
  s = re.sub('-', '', s)
  s = re.sub(':', '', s)
  s = s[:8]

  return datetime.datetime.strptime(s, format)

## convert to datetime
def ConvertToDateTime(x, dateTimeFormat="%Y-%m-%d %H:%M:%S", strLen=19):

  if (x == None) or (x == 'nan') or (x == np.nan):
    return pd.NaT
  if (type(x).__name__ == 'float') and math.isnan(x):
    return pd.NaT
  s = str(x)
  if s == 'nan':
    return pd.NaT

  return datetime.datetime.strptime(x[:strLen], dateTimeFormat)

## also lets define a function generator version for easy mapping
def ConvertToDateTimeFcn(dateTimeFormat="%Y-%m-%d %H:%M:%S", strLen=19):

  def F(x):
    return ConvertToDateTime(x, dateTimeFormat=dateTimeFormat, strLen=strLen)

  return F

## convert weekday returned by isoweekday() to string
def WeekDayToStr(x):

    d = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
    if x in d.keys():
      return(d[x])

    return 'nan'

'''
x = datetime.datetime(2017, 01, 01)
u = x.isoweekday()
WeekDayToStr(u)
'''

## integrates columns (integCols) out from a data frame.
# It uses integFcn for integration
# it only keeps valueCols for integration
def IntegOutDf(df, integFcn, integOutCols, valueCols=None):

  cols = list(df.columns)

  if valueCols == None:
    valueCols = list(filter(lambda x: x not in integOutCols, cols))

  gCols = list(filter(lambda x: x not in (integOutCols + valueCols), cols))
  if len(gCols) > 0:
    cols = gCols + valueCols
  df = df[cols]

  if len(gCols) == 0:
    gCols = ['tempCol']
    df['tempCol'] = 1

  g = df.groupby(by=gCols)
  dfAgg = g.aggregate(integFcn)
  dfAgg = dfAgg.reset_index()

  if 'tempCol' in dfAgg.columns:
    del dfAgg['tempCol']

  return dfAgg

'''
size =  10
df = pd.DataFrame({'categ1':np.random.choice(a=['a', 'b', 'c', 'd', 'e'], size=size, replace=True),
                    'categ2':np.random.choice(a=['A', 'B'], size=size, replace=True),
                    'col1':np.random.uniform(low=0.0, high=100.0, size=size),
                    'col2':np.random.uniform(low=0.0, high=100.0, size=size)
                    })
print(df)
IntegOutDf(df, integFcn=sum, integOutCols=['categ1'], valueCols=['col1', 'col2'])
'''

## aggregates df with different agg fcns for multiple columns
# gCols is not needed since it will be assume to be
# (all columns - the columns being aggregated)
def AggWithDict(df, aggDict, gCols=None):

  cols = list(df.columns)
  valueCols = aggDict.keys()

  if gCols == None:
    gCols = list(filter(lambda x: x not in (valueCols), cols))

  g = df.groupby(gCols)
  dfAgg = g.aggregate(aggDict)
  dfAgg = dfAgg.reset_index()

  return dfAgg

'''
size =  10
df = pd.DataFrame({'categ1':np.random.choice(a=['a', 'b', 'c', 'd', 'e'], size=size, replace=True),
                    'categ2':np.random.choice(a=['A', 'B'], size=size, replace=True),
                    'col1':np.random.uniform(low=0.0, high=100.0, size=size),
                    'col2':np.random.uniform(low=0.0, high=100.0, size=size)
                    })

df = df.sort_values(['categ2', 'categ1', 'col1', 'col2'])

print(df)
aggDf0 = AggWithDict(df=df, aggDict={'col1':sum, 'col2':min})
aggDf1 = AggWithDict(df=df, aggDict={'col1':sum, 'col2':min}, gCols=['categ2'])
print(aggDf0)
print(aggDf1)
'''

## find rows which have repeatted values on some cols
def FindRepRows(df, cols):
  return pd.concat(g for _, g in df.groupby(cols) if len(g) > 1)

## slice df by sliceCol and with given values in sliceValues
def DfSliceDict(df, sliceCol, sliceValues=None):

  if sliceValues == None:
    sliceValues = list(set(df[sliceCol].values))
  dfDict = {}
  for i in range(len(sliceValues)):
    v = sliceValues[i]
    dfDict[v] = df[df[sliceCol] == v]

  return dfDict

## merge dfDict
def MergeDfDict(dfDict, onCols, how='outer', naFill=None):

  keys = dfDict.keys()
  for i in range(len(keys)):
    key = keys[i]
    df0 = dfDict[key]
    cols = list(df0.columns)
    valueCols = list(filter(lambda x: x not in (onCols), cols))
    df0 = df0[onCols + valueCols]
    df0.columns = onCols + [(s + '_' + key) for s in valueCols]

    if i == 0:
      outDf = df0
    else:
      outDf = pd.merge(outDf, df0, how=how, on=onCols)

  if naFill != None:
    outDf = outDf.fillna(naFill)

  return outDf

'''
def GenDf(size):
  df = pd.DataFrame({'categ1':np.random.choice(a=['a', 'b', 'c', 'd', 'e'], size=size, replace=True),
                      'categ2':np.random.choice(a=['A', 'B'], size=size, replace=True),
                      'col1':np.random.uniform(low=0.0, high=100.0, size=size),
                      'col2':np.random.uniform(low=0.0, high=100.0, size=size)
                      })
  df = df.sort_values(['categ2', 'categ1', 'col1', 'col2'])
  return(df)
size = 5
dfDict = {'US':GenDf(size), 'IN':GenDf(size), 'GER':GenDf(size)}
MergeDfDict(dfDict=dfDict, onCols=['categ1', 'categ2'], how='outer', naFill=0)
'''

## split data based on values of a column: col
def SplitDfByCol(df, col):
  #create unique list of device names
  uniqueNames = df[col].unique()
  #create a data frame dictionary to store your data frames
  dfDict = {elem : pd.DataFrame for elem in uniqueNames}
  for key in dfDict.keys():
    dfDict[key] = df[df[col] == key]
  return dfDict

## calculates  value_counts() aka freq for combination of cols
def CombinFreqDf(df, cols=None, countColName='cnt', propColName='prop (%)'):

  if cols == None:
    cols = list(df.columns)
  if (len(cols) < 2):
    cols.append('dummy')
    df['dummy'] = 'NA'
  outDf = df[cols].groupby(cols).agg(len).reset_index()
  outDf.columns = list(outDf.columns[:len(outDf.columns)-1]) + [countColName]
  outDf[propColName] = 100.0 * outDf[countColName] / outDf[countColName].sum()
  outDf = outDf.sort_values([countColName], ascending=[0])

  if ('dummy' in cols):
    del outDf['dummy']
  outDf = outDf.reset_index(drop=True)

  return outDf

'''
df0 = pd.DataFrame({'app':['fb', 'fb', 'randomMail', 'randomMail'], 'party':['1P', '1P', '3P', '3P']})
CombinFreqDf(df=df0, cols=['app', 'party'])
'''

## this function works on a data frame with two categorical columns
## one is the  category column
## one is the label column
## for each category it creates a distribution for the labels
def CalcFreqTablePerCateg(df, categCol, valueCol):

  def AggFcnBuild(categValue):
    def F(x):
      return sum(x == categValue)/(1.0)
    return(F)

  df = df.fillna('NA')
  labels = list(set(df[valueCol]))

  def G(value):
    AggFcn = AggFcnBuild(value)
    dfAgg = df.groupby([categCol])[[valueCol]].agg(lambda x: AggFcn(x))
    dfAgg = dfAgg.reset_index()
    return dfAgg

  value = labels[0]
  dfAgg = G(value)
  for i in range(1, len(labels)):
    value = labels[i]
    dfAgg1 = G(value)
    dfAgg = pd.merge(dfAgg, dfAgg1, how='left', on=[categCol])
  dfAgg.columns = [categCol] + labels
  outDict = {'df': dfAgg, 'labels': labels}

  return outDict

'''
size = 20
df0 = pd.DataFrame({'categ':np.random.choice(a=['a', 'b', 'c'], size=size, replace=True),
                    'value':np.random.choice(a=['AA', 'BB', 'CC'], size=size, replace=True),
                    'col2':np.random.uniform(low=0.0, high=100.0, size=size),
                    'col3':np.random.uniform(low=0.0, high=100.0, size=size),
                    'col4':np.random.uniform(low=0.0, high=100.0, size=size)})

CalcFreqTablePerCateg(df=df0, categCol='categ', valueCol='value')['df']
'''

##  merges a dict of tables
def MergeTablesDict(tabDict):

  keys = tabDict.keys()
  #print(keys)
  n = len(keys)
  for i in range(n):
    key = keys[i]
    tab = tabDict[key]
    df = PropDfTab(tab)
    df = df[['categ', 'freq', 'prop']]
    df.columns = ['categ', 'freq_' + key, 'prop_' + key]
    if i == 0:
      outDf = df
    else:
      outDf = pd.merge(outDf, df, on=['categ'], how='outer')
      outDf = outDf.reset_index(drop=True)
  outDf = outDf.fillna(value=0)

  return outDf

## creating a single string column using multiple columns (cols)
# and adding that to the data frame
def ConcatColsStr(df, cols, colName=None, sepStr='-'):

  x = ''
  if colName == None:
    colName = sepStr.join(cols)
  for i in range(len(cols)):
    col = cols[i]
    x = (x + df[col].map(str))
    if (i < len(cols)-1):
      x = x +'-'
  df[colName] = x

  return(df)

'''
df = pd.DataFrame({'a':range(3), 'b':['rr',  'gg', 'gg'], 'c':range(3)})
ConcatColsStr(df=df, cols=['a', 'b', 'c'], colName=None, sepStr='-')
'''

## flatten a column (listCol) of df with multiple values
def FlattenDfRepField(df, listCol, sep=None):

  if sep != None:
    df = df.assign(**{listCol: df[listCol].str.split(',')})
  outDf = pd.DataFrame({
      col: np.repeat(df[col].values, df[listCol].str.len())
      for col in df.columns.difference([listCol])
  }).assign(
      **{listCol: np.concatenate(df[listCol].values)})[df.columns.tolist()]

  return outDf

'''
df = pd.DataFrame({'var1': ['a,b,c', 'd,e,f'], 'var2': [1, 2], 'var3':[5, 6]})
print(df)
FlattenDfRepField(df, listCol='var1', sep=',')
'''

### tables p-value
## for a given table of frequencies and for each category calculates
## the total count of other categs (complement)
def TabCategCompl(tab, categCol, freqCol, complementCol=None):

  categs = tab[categCol].values
  s = tab[freqCol].sum()
  complement = list()

  for i in range(tab.shape[0]):
    categ = categs[i]
    tab0 = tab[(tab[categCol] == categ)]
    x = tab0[freqCol].values[0]
    c = s - x
    complement.append(c)

  if complementCol == None:
    complementCol = freqCol + '_compl'
  tab[complementCol] = complement

  return tab

## does above for multiple columns
def TabCategComplMulti(tab, categCol, freqCols):

  complementCols = []

  for i in range(len(freqCols)):
    freqCol = freqCols[i]
    tab = TabCategCompl(tab=tab, categCol=categCol, freqCol=freqCol,
                        complementCol=None)
    complementCol = freqCol + '_compl'
    complementCols.append(complementCol)

  cols = freqCols + complementCols
  tab = tab[[categCol] + cols]

  return tab

## adds a p-value per categ for comparing two frequencies
def TabComplPvalue(tab, categCol, freqCols):

  tab = TabCategComplMulti(tab, categCol, freqCols)
  n = tab.shape[0]
  pvalueList = []

  for i in range(n):
    r = tab.iloc[i]
    d = pd.DataFrame({'col1': [r[1], r[2]], 'col2': [r[3],r[4]]})
    pvalue = scipy.stats.fisher_exact(table=d, alternative='two-sided')[1]
    pvalueList.append(Signif(3)(pvalue))

  tab['p-value'] = pvalueList

  return tab

#### Useful functions for mapping a string column to another string column
## using a pattern string
## function for classification mapping
## it uses patterns to map to categories (general purpose)
def LabelByPattern(x, patternDf, patternCol='pattern', categCol='category',
                   exactMatch=False):
  # x a series
  # patternDict has patterns and labels
  # remove duplicate rows
  import re
  patternDf = patternDf.drop_duplicates(keep='first')
  patterns = patternDf[patternCol]
  categs = patternDf[categCol]
  y = ['']*len(x)
  outDf = pd.DataFrame({'x':x, 'y':y})

  for i in range(len(patterns)):
    pattern = patterns[i]
    categ = categs[i]
    hasCateg = x.str.contains(pattern)
    if exactMatch:
      hascateg = (x.str == pattern)

    ind = np.where(hasCateg > 0)[0].tolist()
    for j in ind:
      if not bool(re.search(categ, y[j])):
        y[j] = y[j] + categ

  outDf['y'] = y
  outDf.columns = ['signal', categCol]

  return outDf

## label a data frame based on patternDf
# which includes pattern column and category column
def LabelByPatternDf(df, signalCol, patternDf, patternCol, categCol,
                     newColName='mapped_category'):

  patternDf = patternDf[[patternCol, categCol]]
  patternDf = patternDf.drop_duplicates(keep='first')
  patternDf = patternDf.reset_index(drop=True)
  x = df[signalCol]
  df2 = LabelByPattern(x=x, patternDf=patternDf, patternCol=patternCol,
                       categCol=categCol)
  df[newColName] = df2[categCol]

  return df

######################### Graphical/Plotting Functions #########################
## bar charts for multiple columns (yCols), with different colors
## x axis labels come from the xCol
def BarPlotMultiple(df, xCol, yCols, rotation=45, pltTitle=''):

  x = range(len(df[xCol]))
  colorList = ['r', 'm', 'g', 'y', 'c']
  x = 8*np.array(x)

  for i in range(len(yCols)):
    col = yCols[i]
    x1 = x + 1*i
    plt.bar(x1, df[col], color=colorList[i], alpha=0.6, label=col)

  locs, labels = plt.xticks()
  plt.xticks(x1, df[xCol], rotation='vertical')
  plt.setp(labels, rotation=rotation, fontsize=10)
  plt.title(pltTitle + ': ' + xCol)
  plt.legend()

## make a boxplot for multiple columns Side by Side (Sbs, include mean with a star
def BoxPlotDfColsSbS(df, cols=None, pltTitle='', xlab='', ylab='value',
  boxColors=['darkkhaki', 'royalblue', 'r', 'g', 'y', 'o', 'b'], ylim=None):

  from matplotlib.patches import Polygon
  data = []

  if cols is None:
    cols = df.columns
  for i in range(len(cols)):
    col = cols[i]
    data.append(df[col])

  fig, ax1 = plt.subplots(figsize=(10, 6))
  fig.canvas.set_window_title('')
  plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

  bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
  plt.setp(bp['boxes'], color='black')
  plt.setp(bp['whiskers'], color='black')
  plt.setp(bp['fliers'], color='red', marker='+')
  # Add a horizontal grid to the plot, but make it very light in color
  # so we can use it for reading data values but not be distracting
  ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                 alpha=0.5)

  # Hide these grid behind plot objects
  ax1.set_axisbelow(True)
  ax1.set_title(pltTitle, fontsize=20)
  ax1.set_xlabel(xlab)
  ax1.set_ylabel(ylab)

  # Now fill the boxes with desired colors
  numBoxes = len(data)
  medians = list(range(numBoxes))

  for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[i], label=cols[i])
    ax1.add_patch(boxPolygon)
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
      medianX.append(med.get_xdata()[j])
      medianY.append(med.get_ydata()[j])
      plt.plot(medianX, medianY, 'k')
      medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
             color='w', marker='*', markeredgecolor='k')

  if ylim is not None:
    ax1.set_ylim(ylim)
  plt.legend()

## it plots all columns wrt index
# it uses colors to compare them side by side.
def PlotColumnsWrtIndex(df, cols=None, categCol=None, pltTitle='',
  ymin=None, ymax=None, yLabel='', xLabel='', colorList=None,
  alphaList=None, sciNotation=False, ExtraFcn=None, orient='v',
  sizeAlpha=0.75, legendColAlpha=2):

  df2 = df.copy()

  if cols == None:
    cols = list(df2.columns)

  if categCol is not None:
    df2.index = df2[categCol]
    if (categCol in cols):
      cols = list(set(cols) - set([categCol]))
      #cols = cols.remove(categCol)
      print(cols)
  categs = df2.index
  Mark(categs)
  num = len(categs)
  x = range(num)

  if colorList == None:
    colorList = ['r', 'g', 'm', 'y', 'c', 'darkkhaki', 'royalblue', 'k']

  if alphaList == None:
    alphaList = [0.5] * len(cols)
  stretch = 4 * len(cols)
  x = stretch * np.array(x)

  if orient == 'v':
    fig, ax = plt.subplots(figsize=(15*sizeAlpha, 10*sizeAlpha),
      dpi=1200, facecolor='w', edgecolor='k')

    for i in range(len(cols)):
      col = df2[cols[i]]
      plt.bar(x + 2*i, col.values, alpha=alphaList[i], label=cols[i],
              color=colorList[i], width=2, edgecolor='black',
              linewidth=2.0*sizeAlpha)
    plt.title(pltTitle, fontsize=20, fontweight='bold')

    if sciNotation:
      plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = categs
    ax.set_xticklabels(labels)
    locs, labels = plt.xticks(x, categs)
    plt.setp(labels, rotation=15, fontsize=17*sizeAlpha, fontweight='bold')

    locs2, labels2 = plt.yticks()
    plt.setp(labels2, rotation=0, fontsize=17*sizeAlpha, fontweight='bold')

    ncol = (len(cols)/legendColAlpha)
    plt.legend(
        loc='best',
        fontsize=17,
        prop={'weight': 'semibold',
              'size': 17 * sizeAlpha},
        ncol=ncol)
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if ExtraFcn is not None:
      ExtraFcn(ax)

  if orient == 'h':

    fig, ax = plt.subplots(figsize=(10*sizeAlpha, 15*sizeAlpha), dpi=1200,
      facecolor='black', edgecolor='black')
    for i in range(len(cols)):
      col = df2[cols[i]]
      plt.barh(
          x + 2 * (i - 1),
          col.values,
          alpha=alphaList[i],
          label=cols[i],
          color=colorList[i],
          height=2,
          edgecolor='black',
          linewidth=2.0 * sizeAlpha)
    plt.title(pltTitle, fontsize=20*sizeAlpha, fontweight='bold')
    if sciNotation:
      plt.ticklabel_format(
          style='sci', axis='x', scilimits=(0, 0), prop={'weight': 'bold'})
    labels = categs
    ax.set_yticklabels(labels)
    locs, labels = plt.yticks(x, categs)
    plt.setp(labels, rotation=0, fontsize=17*sizeAlpha, fontweight='bold')

    locs2, labels2 = plt.xticks()
    plt.setp(labels2, rotation=20, fontsize=17*sizeAlpha, fontweight='bold')
    ncol = len(cols) / legendColAlpha

    plt.legend(
        loc='best',
        ncol=ncol,
        prop={'weight': 'semibold',
              'size': 17 * sizeAlpha})
    axes = plt.gca()
    axes.set_xlim([ymin, ymax])
    ax.set_xlabel(yLabel)
    ax.set_ylabel(xLabel)
    if ExtraFcn != None:
      ExtraFcn(ax)
    plt.gca().invert_yaxis()

  return {'fig': fig, 'ax': ax}

'''
df0 = pd.DataFrame({'categ':np.random.choice(a=['aaasfsa', 'bsdafds', 'fafsfsasf'], size=5, replace=True),
                    'col1':np.random.uniform(low=0.0, high=100.0, size=5),
                    'col2':np.random.uniform(low=0.0, high=100.0, size=5),
                    'col3':np.random.uniform(low=0.0, high=100.0, size=5),
                    'col4':np.random.uniform(low=0.0, high=100.0, size=5)})

PlotColumnsWrtIndex(df=df0, cols=['col1', 'col2', 'col3', 'col4'], categCol='categ', orient='v', sciNotation=True)
'''

## this function creates a plot with each bar representing the distribution
# for a category (given in categCol)
# each distribution is defined on a set of labels
# the distributions are given in each column
def PlotStackedDist_perCateg(
    df, categCol, cols=None, labels=None,
    sortCols=None, figsize=(10, 5), mainText=''):

  import colorsys
  if cols == None:
    cols = list(df.columns[1:len(df.columns)])
  if labels == None:
    labels = cols
  if sortCols == None:
    sortCols = cols

  df = df.sort(sortCols, ascending=False)
  m = len(cols)
  HSV_tuples = [(x*1.0/m, 0.5, 0.5) for x in range(m)]
  RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
  n = df.shape[0]
  x = pd.Series(n*[0])
  bar_locations = np.arange(n)
  fig, ax = plt.subplots(figsize=figsize)

  for i in range(len(cols)):
    col = cols[i]
    y = df[col].values
    ax.bar(bar_locations, y, bottom=x, color=RGB_tuples[i], label=labels[i])
    x = x + y

  plt.legend(loc="best")
  bar_locations2 = np.arange(n) + 0.5
  plt.xticks(bar_locations2, df[categCol].values, rotation='vertical')
  plt.title(mainText)
  print(bar_locations)
  print(df.loc[0].values)
  fig.show()

'''
df0 = pd.DataFrame({'country':['JP', 'US', 'FR'],
                    'col1':np.random.uniform(low=0.0, high=100.0, size=3),
                    'col2':np.random.uniform(low=0.0, high=100.0, size=3),
                    'col3':np.random.uniform(low=0.0, high=100.0, size=3),
                    'col4':np.random.uniform(low=0.0, high=100.0, size=3)})


PlotStackedDist_perCateg(
    df=df0, categCol='country', cols=['col1', 'col2', 'col3', 'col4'], labels=None,
    sortCols=None, figsize=(10, 5), mainText='')

'''

## compares the values (valueCol) for the index (pivotIndCol)
# for various classes given in compareCol
# first it pivots the data and then plots side by side
def PivotPlotWrt(
  df, pivotIndCol, compareCol, valueCol, cols=None, pltTitle='', sizeAlpha=0.75):

  dfPivot = df.pivot(index=pivotIndCol, columns=compareCol, values=valueCol)
  dfPivot = dfPivot.fillna(0)
  dfPivot[pivotIndCol] = dfPivot.index

  if cols is None:
    cols = list(set(df[compareCol].values))

  p = PlotColumnsWrtIndex(
      df=dfPivot,
      cols=cols,
      categCol=pivotIndCol,
      orient='h',
      pltTitle=pltTitle,
      sizeAlpha=sizeAlpha)

  return {'df':dfPivot, 'plt':p}

## creating quantiles for a continuous variable and removing repetitions
def Qbins(x, num=10):

  qs = list(
      np.percentile(a=x,
                    q=list(100 * np.linspace(
                        start=0,
                        stop=(1 - 1.0/num),
                        num=num))))
  qs = list(set([float("-inf")] + qs + [float("inf")]))
  qs.sort()

  return qs

# cuts uniformly
def Ubins(x, num=10):

  b = np.linspace(start=min(x), stop=max(x), num=num)
  b = map(Signif(2), b)
  b = list(set(b))
  b = [float("-inf")] + b + [float("inf")]
  b.sort()

  return b

## cuts using quantiles
def CutQ(x, num=10):
  qs = Qbins(x, num)
  discX = pd.cut(x, bins=qs)
  return(discX)

## make a bar plot
def BarPlot(y, yLabels, ylim=None, pltTitle='', figSize=[5, 5]):

  n = len(y)
  x = pd.Series(n*[0])
  bar_locations = np.arange(n)
  fig, ax = plt.subplots()
  fig.set_size_inches(figSize[0], figSize[1])
  ax.bar(bar_locations, y, bottom=x, color='r')
  plt.legend(loc="best")
  bar_locations2 = np.arange(n) + 0.5
  plt.xticks(bar_locations2, yLabels, rotation='vertical')
  axes = plt.gca()
  axes.set_ylim(ylim)
  plt.title(pltTitle)
  fig.show()

## creates a cut column with NA being an explicit category
def ExplicitNa_cutCol(df, col, cuts, newCol=None):

  if newCol is None:
    newCol = col + '_range'

  df[newCol] = pd.cut(df[col], cuts)
  df[newCol] = df[newCol].cat.add_categories(["NA"])
  df[newCol] = df[newCol].fillna("NA")

  return df

'''
z = np.random.normal(loc=50.0, scale=20.0, size=10)
z = np.insert(z, 0, float('nan'))
df0 = pd.DataFrame({'z':z})
ExplicitNa_cutCol(df=df0, col='z', cuts=[-20, 0, 20, 40, 60, 80, 100, 120, 140, float('inf')], newCol=None)
'''

## order df based on a cut column
def OrderDf_cutCol(df, cutCol, orderCol='order'):

  def F(s):
    x = re.search(r'.*?\((.*),.*', s)
    if x is None:
      return(float('-inf'))
    return(float(x.group(1)))

  df['order'] = df[cutCol].map(F)
  df = df.sort_values('order')

  return df

'''
z = np.random.normal(loc=50.0, scale=20.0, size=10)
z = np.insert(z, 0, float('nan'))
df0 = pd.DataFrame({'z':z})
u = pd.cut(z, [-20, 0, 20, 40, 60, 80, 100, 120, 140, float('inf')])
df0['col'] = u
df0['col'] = df0['col'].cat.add_categories(["NA"])
df0['col'] = df0['col'].fillna("NA")
OrderDf_cutCol(df=df0, cutCol="col")
'''

## for a variable generated with pd.cut it make a barplot
# it orders the labels based on the their values (rather than freq)
def FreqPlot_cutCol(u, pltTitle='', figSize=[5, 5]):

  tab = u.value_counts()
  df0 = pd.DataFrame(tab)
  df0['label'] = list(df0.index)
  df0 = OrderDf_cutCol(df=df0, cutCol='label')
  df0.columns = ['value', 'label', 'order']
  df0 = df0.sort_values('order')
  BarPlot(y=df0['value'], yLabels=df0['label'], pltTitle=pltTitle, figSize=figSize)

  return df0

'''
z = np.random.normal(loc=50.0, scale=20.0, size=1000)
u = pd.cut(z, [-20, 0, 20, 40, 60, 80, 100, 120, 140, float('inf')])
FreqPlot_cutCol(u)
'''

def PropDfTab(tab, ylim=None, categCol='categ', plotIt=False, pltTitle=''):

  d = pd.DataFrame(tab)
  d.columns = ['freq']
  e = (100.0 * d.values) / sum(d.values)
  e = map(Signif(5), e)
  d['prop'] = e
  d[categCol] = d.index

  if plotIt:
    BarPlot(y=e, yLabels=list(d.index), ylim=ylim, pltTitle=pltTitle)

  return d

## cut continuous var
def CutConti(x, num=10, method='quantile'):

  if method == 'quantile':
    b = Qbins(x, num=num)
  elif (method == 'uniform'):
    b = Ubins(x, num=num)
  z = pd.cut(x, bins=b)

  return z

## gets a continuous var x, partitions the real line based on quantiles or bins of x
# then generates a function which assigns levels to any new value/values
def CutContiFcn(
    x,
    num=10,
    method='quantile',
    intervalColName='interval',
    levelColName='level',
    levelsType='int',
    levelPrefix='Lev',
    rawValueColName='raw'):

  if method == 'quantile':
    b = Qbins(x, num=num)
    print(b)
  elif (method == 'uniform'):
    b = Ubins(x, num=num)

  intervals = sorted(set(pd.cut(x + b[1:], bins=b)))

  if ():
    levels = map(lambda x: levelPrefix + str(x), range(len(intervals)))

  Mark(levels)
  levDf = pd.DataFrame({intervalColName:intervals, levelColName:levels})
  Mark(levDf)

  def F(u):
    z = pd.cut(u, bins=b)
    df0 = pd.DataFrame({rawValueColName:u, intervalColName: z})
    outDf = pd.merge(df0, levDf, on=[intervalColName], how='left')
    return(outDf)

  return F

'''
x = [1, 3, 4, 5, 66, 77, 88]
F = CutContiFcn(x, num=10, method='quantile', intervalColName='interval',
                levelColName='level', levelPrefix='Lev', rawValueColName='raw')
F(x)
F(x + [5, 1, 3, 100, -1 , 90, 2.2])
'''

## cuts continuous data and creates a bar plot
def CutBarPlot(x, num=10, method='quantile', pltTitle='', figSize=[5, 5]):

  z = CutConti(x, num=num, method=method)
  u = z.value_counts()
  #print(u)
  d = pd.DataFrame(u)
  d = 100.0 * (d / d.sum())
  d = d.sort_index()
  BarPlot(y=d.values, yLabels=d.index, ylim=None, pltTitle=pltTitle,
          figSize=figSize)

## returns a functions which calculates quantiles according to q (which )
def QuantileFcn(q):

  def F(x):
    return(np.percentile(a=x, q=q))

  return F

def PlotQuantilesPerSlice(df, sliceCol, respCol, gridNum=100.0, pltTitle=''):

  slices = list(set(df[sliceCol].values))
  outDict = {}

  for sl in slices:
    x = df[df[sliceCol] == sl][respCol].values
    grid = list(
        gridNum * np.linspace(
            start=1.0/gridNum,
            stop=(1 - 1.0/gridNum),
            num=gridNum))
    q = QuantileFcn(grid)(x)
    outDict[sl] = q
    plt.plot(grid, q, label=sl)

  plt.legend()
  plt.title(pltTitle)
  outDf = pd.DataFrame(outDict)

  return outDf

'''
df = pd.DataFrame({
  'value':[1, 1, 1, 2, 3, 4, 2, 2, 2],
  'categ':['a', 'a', 'b', 'b', 'b', 'a', 'a', 'b', 'a']})
PlotQuantilesPerSlice(df=df, sliceCol='categ', respCol='value', pltTitle='')
'''

## takes a vector of labels, eg pandas series
# it returns a freq table with props in dataframe format
def GenFreqTable(x, rounding=None):

  freqTab = x.value_counts()
  distbnTab = 100.0 * x.value_counts() / freqTab.sum()
  labels = freqTab.keys()
  freqValues = list(freqTab)
  propValues = list(distbnTab)

  if rounding is not None:
    propValues = map(Signif(rounding), propValues)

  outDict = {'label':labels, 'freq':freqValues, 'prop':propValues}
  outDf = pd.DataFrame(outDict)
  outDf = outDf[['label', 'freq', 'prop']]

  return outDf

'''
x = pd.Series(['a', 'a', 'b', 'b', 'c'])
print(GenFreqTable(x))
'''

## builds a categ distbn for each combination after groupby indCols
def CategDistbnDf(df, indCols, categCol, rounding=None):

  def F1(x):
    return tuple(GenFreqTable(x)['label'].values)

  def F2(x):
    return tuple(GenFreqTable(x)['freq'].values)

  def F3(x):
    return tuple(GenFreqTable(x, rounding=4)['prop'].values)

  df = df[indCols + [categCol]].copy()

  df[categCol + '_freq'] = df[categCol]
  df[categCol + '_prop'] = df[categCol]
  g = df.groupby(indCols)
  outDf = g.aggregate({categCol:F1 , categCol + '_freq':F2,
                       categCol + '_prop':F3})
  outDf = outDf.reset_index()
  outDf2 = outDf[BringElemsToFront(outDf.columns, indCols + [categCol])]

  return outDf2

'''
df = pd.DataFrame({'user_id':[1, 1, 1, 2, 2, 2, 2, 2, 1, 1],
                   'interface':['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B'],
                   'categ':['a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'b', 'b']})
dnDf = CategDistbnDf(df=df, indCols=['user_id', 'interface'], categCol='categ', rounding=None)
'''

## compare label distribution across slices
def LabelDistbn_acrossSlice(
  df,
  sliceCol,
  labelCol,
  plotIt=True,
  pltTitle='',
  orderIntervals=False,
  sortBySlice=False):

  def F(group):
    return CombinFreqDf(group[[sliceCol, labelCol]])

  g = df.groupby([sliceCol], as_index=False)
  outDf = g.apply(F).reset_index(drop=True)
  slices = list(set(df[sliceCol].values))
  horizDf = None
  for s in slices:
    s = str(s)
    subDf = outDf[outDf[sliceCol].map(str) == s][[labelCol, 'cnt', 'prop (%)']]
    subDf.columns = [labelCol, s + '_cnt', s + '_prop (%)']
    if horizDf is None:
      horizDf = subDf
      horizDf['total_cnt'] = subDf[s + '_cnt']
    else:
      horizDf = pd.merge(horizDf, subDf, on=labelCol, how='outer')
      horizDf['total_cnt'] = horizDf['total_cnt'] + horizDf[s + '_cnt']

  horizDf = horizDf.sort_values(['total_cnt'], ascending=[0])
  if orderIntervals:
    horizDf = OrderDf_cutCol(df=horizDf, cutCol=labelCol, orderCol='order')
  if sortBySlice:
    horizDf.sort_values([sliceCol])

  horizDf = horizDf[:20]
  p = None

  if plotIt:
    p = PlotColumnsWrtIndex(
      df=horizDf,
      cols=map(lambda x: str(x) + '_prop (%)', slices),
      categCol=labelCol,
      orient='h',
      pltTitle=pltTitle)

  return {'outDf':outDf, 'horizDf':horizDf, 'p':p}

'''
df = GenUsageDf_forTesting()
Mark(df[:2])

res = LabelDistbn_acrossSlice(df=df, sliceCol='expt', labelCol='prod',
                              plotIt=True)
res['p']
'''
# make a single label distbn
def LabelDistbn(
    df,
    labelCol,
    orderIntervals=False,
    pltTitle="",
    CustomOrder=None,
    figSize=[10, 8]):

  out = CombinFreqDf(df[[labelCol]])
  del out['cnt']
  out['prop (%)'] = out['prop (%)'].map(Signif(3))

  if orderIntervals:
    out = OrderDf_cutCol(df=out, cutCol=labelCol, orderCol='order')
    del out['order']

  if CustomOrder is not None:
    out = CustomOrder(out)

  fig, ax = plt.subplots();
  fig.set_size_inches(figSize[0], figSize[1])

  plt.bar(range(len(out)), out['prop (%)'])
  plt.xticks(np.array(range(len(out))) + 0.5, out[labelCol], rotation=90)
  plt.grid(False)
  plt.grid(axis='y',  linewidth=1, color='red', alpha=0.5)
  if pltTitle == "":
    pltTitle = labelCol + " distbn"
  plt.title(pltTitle, fontsize=20, fontweight='bold')

  return(out)

## make a label distbn for each slice
def LabelDistbn_perSlice(
  df,
  sliceCol,
  labelCol,
  plotIt=True,
  pltTitle='',
  orderIntervals=False,
  sortBySlice=False,
  labels=None,
  sizeAlpha=0.75):

  def F(group):
    return CombinFreqDf(group[[sliceCol, labelCol]])

  g = df.groupby([labelCol], as_index=False)
  outDf = g.apply(F).reset_index(drop=True)

  if labels is None:
    labels = list(set(df[labelCol].values))

  horizDf = None
  for l in labels:
    l = str(l)
    subDf = outDf[outDf[labelCol].map(str) == l][[sliceCol, 'cnt', 'prop (%)']]
    subDf.columns = [sliceCol, l + '_cnt', l + '_prop (%)']

    if horizDf is None:
      horizDf = subDf
      horizDf['total_cnt'] = subDf[l + '_cnt']
    else:
      horizDf = pd.merge(horizDf, subDf, on=sliceCol, how='outer')
      horizDf = horizDf.fillna(0)
      horizDf['total_cnt'] = horizDf['total_cnt'] + horizDf[l + '_cnt']

  horizDf = horizDf.sort_values(['total_cnt'], ascending=[0])
  if orderIntervals:
    horizDf = OrderDf_cutCol(df=horizDf, cutCol=sliceCol, orderCol='order')

  if sortBySlice:
    horizDf = horizDf.sort_values([sliceCol])

  horizDf = horizDf[:20]
  p = None

  for l in labels:
    horizDf[l + '_%'] = 100 * (horizDf[l + '_cnt'] / horizDf['total_cnt'])

  if plotIt:
    p = PlotColumnsWrtIndex(
      df=horizDf,
      cols=map(lambda x: str(x) + '_%', labels),
      categCol=sliceCol,
      orient='h',
      pltTitle=pltTitle,
      sizeAlpha=sizeAlpha)

  return {'outDf':outDf, 'horizDf':horizDf, 'p':p}

## interpolate missing categ values wrt certain columns
# condDict will determine which subset of data should be used to make predictions
# replacedValues are the values which need replacement/interpolation
# dropUnassigned is to determine if we should keep rows which remain unassigned around
def InterpCategColWrt(
    df, yCol, xCols, condDict={}, replacedValues=None, dropUnassigned=True):

  df2 = df.copy()
  if len(condDict) > 0:
    ind = BuildCondInd(df=df2, condDict=condDict)
    df2 = df2[ind].copy()
  if replacedValues is not None:
    df2 = df2[~df2[yCol].isin(replacedValues)].copy()

  predDf = CategDistbnDf(df=df2, indCols=xCols, categCol=yCol)
  predDf[yCol + '_pred'] = predDf[yCol].map(lambda x: x[0])
  ind = df[yCol].isin(replacedValues)
  df3 = df[ind].copy()
  df4 = pd.merge(df3, predDf[xCols + [yCol + '_pred']], on=xCols, how='left')
  df.loc[ind, yCol] = df4[yCol + '_pred'].values

  if dropUnassigned:
    df = df.dropna(subset=[yCol])

  return df

'''
df = pd.DataFrame({'user_id':[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3],
                   'os':['and', 'and', 'and', 'ios', 'ios', 'ios', 'ios', 'and', 'and', 'and', 'ios', 'ios', 'ios', 'ios', 'ios'],
                   'y':[None, 'c', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'b', 'b', 'nan', 'b', 'b', None],
                   'country': ['us', 'us', 'us', 'us', 'us', 'jp', 'us', 'jp', 'us', 'us', 'us', 'us', 'jp', 'us', 'us']})
print(df)
InterpCategColWrt(df=df,
                  yCol='y',
                  xCols=['user_id', 'os'],
                  condDict={'country':['us']},
                  replacedValues=['nan', None],
                  dropUnassigned=False)
'''

## for a df with sliceCols, it groups by sliceCols and for each categCols combination,
# it adds a total count column for the valueCol
# for example sliceCols: country, categCols=[sequence, event_1st, event_2nd]
# and valueCol=sequence_count
# we can figure out the total frequency of each sequence in each country
#  as well as the frequency of the first event for the same country (sliceCols)
# we also agg a grand total for the valueCols for each combination of sliceCols
def AddTotalsDf(df, categCols, valueCols, sliceCols=[], aggFnDict=sum,
                integOutOther=False):

  ## if there are no sliceCols, we generate a tempCol to be sliceCol
  ## then we delete it at the end
  l = len(sliceCols)
  if l == 0:
    sliceCols = ['tempCol']
    df['tempCol'] = 1

  ## integrates out wrt sliceCols + categCols first.
  ## so other columns will be dropped
  ## and the valueCols will be integrated out across,
  ## when there are repeated sliceCols + categCol even if there is no extra col
  if integOutOther:
    df = df[sliceCols + categCols + valueCols]
    g = df.groupby(sliceCols + categCols)
    df = g.agg(aggFnDict)
    df = df.reset_index()

  df0 = df[sliceCols + categCols + valueCols]
  outDf = df.copy()

  for categCol in categCols:
    g = df0.groupby(sliceCols + [categCol])
    aggDf = g.agg(aggFnDict)
    aggDf= aggDf.reset_index()
    aggDf.columns = (sliceCols +
                     [categCol] +
                     map(lambda x: categCol + '_' + x + '_agg', valueCols))
    outDf = pd.merge(outDf, aggDf, on=sliceCols + [categCol], how='left')

  # add slice (sliceCols slice) totals: same as above but we drop the categCol
  df0 = df[sliceCols + valueCols]
  g = df0.groupby(sliceCols)
  aggDf = g.agg(aggFnDict)
  aggDf= aggDf.reset_index()
  aggDf.columns = sliceCols + map(lambda x: x + '_slice_total', valueCols)
  outDf = pd.merge(outDf, aggDf, on=sliceCols, how='left')

  # reorder the columns
  cols = (sliceCols +
          sorted(categCols) +
          valueCols +
          list(sorted(set(outDf) - set(sliceCols + categCols + valueCols))))
  outDf = outDf[cols]
  ## remove extra column if it was created
  if l == 0:
    del outDf['tempCol']

  return outDf

'''
df = pd.DataFrame({'country':['JP', 'JP', 'JP', 'BR', 'BR', 'BR', 'JP', 'JP', 'JP', 'BR'],
                   'sequence':['a>b', 'a>b', 'b>a', 'b>a', 'a>b', 'a>b', 'a>c', 'a>c', 'b>c', 'c>b'],
                   '1st':['a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'b', 'c'],
                   '2nd':['b', 'b', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'b'],
                   'count':[10, 11, 1, 20, 2, 2, 2, 200, 1, 1],
                   'utility':[-10, -11, 1, 20, 2, 2, 2, -200, 1, 1],})

sliceCols = ['country']
categCols = ['sequence', '1st', '2nd']
valueCols = ['count', 'utility']
aggFnDict = {'count':sum, 'utility':np.mean}
AddTotalsDf(df=df, categCols=categCols, valueCols=valueCols, sliceCols=sliceCols, aggFnDict=sum)
AddTotalsDf(df=df, categCols=categCols, valueCols=valueCols, sliceCols=sliceCols, aggFnDict=aggFnDict)
AddTotalsDf(df=df, categCols=categCols, valueCols=valueCols, sliceCols=sliceCols, aggFnDict=aggFnDict, integOutOther=True)
AddTotalsDf(df=df, categCols=categCols, valueCols=valueCols, sliceCols=[], aggFnDict=aggFnDict, integOutOther=True)
'''

## for a data frame with a countCol, we do bootstrap
def BsWithCounts(df, countCol=None):

  if countCol == None:
    n = df.shape[0]
    ind = np.random.choice(a=n, size=n, replace=True, p=None)
    df2 = df.iloc[ind]
    return(df2)

  df = df.reset_index(drop=True)
  rowInd = list(range(len(df)))
  counts = df[countCol].values
  longInd = []

  for a, b in zip(rowInd, counts):
      longInd.extend([a] * b)

  bsLongInd = np.random.choice(a=longInd, size=len(longInd), replace=True, p=None)
  bsIndDf = pd.DataFrame(pd.Series(bsLongInd).value_counts())
  bsRowInd = list(bsIndDf.index)
  bsCounts = bsIndDf[0].values
  df2 = df.iloc[bsRowInd]
  df2[countCol] = bsCounts
  df2 = df2.reset_index(drop=True)

  return df2

'''
df = pd.DataFrame({'a':['cats', 'horses', 'dogs', 'wolves'], 'count':[2, 10, 4, 1]})
Mark(df, 'original df')
countCol = 'count'
Mark(BsWithCounts(df, countCol), ' using counts')
Mark(BsWithCounts(df, countCol=None), ' not using counts')
'''

## get a sublist with unique elements
# while preserving order
def UniqueList(l):
  seen = set()
  seen_add = seen.add
  return [x for x in l if not (x in seen or seen_add(x))]

'''
x = [1, 2, 1, 1, 2, 3] + range(100000) + [1, 2, 1, 1, 2, 3]

tic = time.clock()
UniqueList(x)
toc = time.clock()
Mark((toc-tic)*100)

tic = time.clock()
set(x)
toc = time.clock()
Mark((toc-tic)*100)
'''

## bring certain elements (x) of a list (l) to front
# without re-ordering others
def BringElemsToFront(l, subList):

  front = []
  for k in range(len(subList)):
    front = front + [j for i,j in enumerate(l) if j == subList[k]]
  end = [j for i,j in enumerate(l) if not (j in subList)]

  return front + end

'''
BringElemsToFront(l=[1, 2, 3, 1], subList=[1])
BringElemsToFront(l=[1, 2, 3, 1, 4, 5,], subList=[1, 4, 5])
'''
## get a fcn which returns a color grid of size n
def GetColorGridFcn(n):

    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = matplotlib.colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color

'''
def main():
    n = 5
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.axis('scaled')
    ax.set_xlim([ 0, n])
    ax.set_ylim([-0.5, 0.5])
    cmap = GetColorGridFcn(n)
    for i in range(n):
        col = cmap(i)
        rect = plt.Rectangle((i, -0.5), 1, 1, facecolor=col)
        ax.add_artist(rect)
    ax.set_yticks([])
    plt.show()

if __name__=='__main__':
    main()
'''

## takes a dictionary to a string
def DictOfListsToString(
  d,
  dictElemSepr='__',
  listElemSepr='_',
  keyValueSepr=':',
  noneString=''):

  if d == None or d == {}:
    return(noneString)
  keys = d.keys()
  keys = map(str, keys)
  keys.sort()

  out = ''
  for key in keys:
    if (d[key] != None):
      l = map(str, d[key])
      value = str(listElemSepr.join(l))
      if out != '':
        out = out + dictElemSepr
      out = out + key + keyValueSepr + value

  return out

'''
d = {'z':[2], 'd':[1], 'e':[2]}
DictOfListsToString(d)

d = {'z':[2], 'd':[1], 'e':None}
DictOfListsToString(d)

condDict = {'form_factor':['PHONE']}
condDict = {'form_factor':None}
condDict = {'form_factor':['PHONE'], 'country':['JP']}
condDict = None
condStr = DictOfListsToString(condDict, dictElemSepr='__', listElemSepr='_')
'''

## plotting confidence intervals given in each row
# will label the rows in labelCol is given
def PlotCI(df, colUpper, colLower, y=None, col=None, ciHeight=0.5,
           color='grey', labelCol=None, pltLabel=''):

  if y == None:
    y = range(len(df))

  minCiWidth = (df[colUpper] - df[colLower]).min()

  if col != None:
    ## following was troubling in log scale,
    # the width of the lines were changing in visualization (not desired)
    '''
    p = plt.barh(
        bottom=y,
        width=np.array([minCiWidth]*len(y)),
        left=df[col],
        height = ciHeight,
        color='green',
        alpha=1,
        label=None)
    '''
    for i in range(len(y)):
      plt.plot(
        [df[col].values[i],
        df[col].values[i]],
        [y[i], y[i] + ciHeight],
        color=color,
        linestyle='-',
        alpha=0.4,
        linewidth=2)

  p = plt.barh(
      bottom=y,
      width=df[colUpper]-df[colLower],
      left=df[colLower],
      color=color,
      height=ciHeight,
      alpha=0.4,
      label=pltLabel);

  if labelCol != None:
    plt.yticks(y, df[labelCol].values, rotation='vertical');

'''
df0 = pd.DataFrame({'med':[1, 2, 3, 10], 'upper':[2, 5, 6, 12],
                    'lower':[-1, -2, -3, 4], 'categ':['a', 'b', 'c', 'd']})
PlotCI(df=df0, colUpper='upper', colLower='lower', y=None, col='med',
       ciHeight=0.5, color='grey', labelCol='categ', pltLabel='')
'''

## compares the CI's for available labels in labeCol
# we do that for each slice with different color to compare
def PlotCIWrt(df, colUpper, colLower, sliceCols, labelCol, col=None,
              ciHeight=0.5, rotation = 0, addVerLines=[], logScale=False,
              lowerLim=None, pltTitle='', figSize=[5, 20]):

  df2 = ConcatColsStr(
    df=df.copy(),
    cols=sliceCols,
    colName='slice_comb',
    sepStr='-')

  labelSet = UniqueList(df2[labelCol].values)
  labelIndDf = pd.DataFrame({labelCol: labelSet})
  labelIndDf = labelIndDf.sort_values([labelCol])
  labelIndDf['labelInd'] = range(len(labelSet))

  n = len(labelIndDf)
  ## groupby each slice
  slicesSet = set(df2['slice_comb'])
  g = df2.groupby(['slice_comb'])
  sliceNum = len(g)
  sliceNames = g.groups.keys()
  sliceNames.sort()
  ColorFcn = GetColorGridFcn(sliceNum + 2)

  plt.figure(1);
  fig, ax = plt.subplots();
  fig.set_size_inches(figSize[0], figSize[1]*(n/20.0))

  for i in range(sliceNum):
    sliceName = sliceNames[i]
    df3 = g.get_group(sliceName)
    df3 = pd.merge(df3, labelIndDf, on=[labelCol], how='outer')
    df3 = df3.sort_values([labelCol])
    df3 = df3.fillna(0)

    ciHeight = 1.0 / sliceNum
    shift = ciHeight * i
    y = map(lambda x: (float(x) + shift), range(n))

    assert (len(df3) == len(y)),("len(y) must be the same as merged df (df3)." +
                                 " This might be because of repeatted rows in df3")

    PlotCI(df=df3, colUpper=colUpper, colLower=colLower, y=y, col=col,
           ciHeight=ciHeight, color=ColorFcn(i + 1), labelCol=labelCol,
           pltLabel=sliceName)

  for j in range(n + 1):
    plt.axhline(y=j, color='grey', alpha=0.5)

  labels = [item.get_text() for item in ax.get_xticklabels()]
  labels = list(labelIndDf[labelCol].values)
  ax.set_yticklabels(labels)
  locs, labels = plt.yticks(map(lambda x: (float(x) + 0.5), range(n)), labels)
  plt.setp(labels, rotation=rotation, fontweight='bold', fontsize="large")

  for x in addVerLines:
    plt.axvline(x=x, color='orange', alpha=0.5)

  if logScale:
    plt.xscale('log')

  if len(addVerLines) > 0:
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #ax.set_xticklabels(map(str, addVerLines))
    ax = plt.gca() # grab the current axis
    ax.set_xticks(addVerLines) # choose which x locations to have ticks
    ax.set_xticklabels(addVerLines) # set the labels to display at those ticks

  #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
  #       ncol=2, mode="expand", borderaxespad=0.)
  plt.legend(
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
    prop={'weight':'bold', 'size':'large'})
  plt.xlim((lowerLim, None))
  plt.xlim((lowerLim, None))
  plt.title(
      pltTitle, fontname="Times New Roman",fontweight="bold",
      fontsize="x-large")

  return fig
  #plt.show()

'''
## Example
df0 = pd.DataFrame({
    'med':[1, 2, 3, 10, 11, 12, 1, 2],
    'upper':[2, 5, 6, 12, 13, 16, 5, 6],
    'lower':[-1, -2, -3, 4, 2, 2, 1, 2],
    'categ':['a', 'b', 'c', 'd', 'a', 'c', 'd', 'a'],
    'country':['JP', 'JP', 'JP', 'US', 'US', 'US', 'BR', 'BR']})

res = PlotCIWrt(
  df=df0,
  colUpper='upper',
  colLower='lower',
  sliceCols=['country'],
  labelCol='categ',
  col='med',
  ciHeight=0.5,
  rotation = 0,
  pltTitle="WTF is going on?",
  figSize=[10, 30])
'''

## this function will partition a df using keyCol
# for which their row (so maybe involve other columns to check conditions)
# satisfy conditions
# any combination of keys which passes the condition at least once will be
# considered as satisfy
def PartDfByKeysWrtCond(df, keyCols, condDict, passColName='passesCond'):

  keyDfUnique = df[keyCols].drop_duplicates()
  ind = BuildCondInd(df=df, condDict=condDict)
  passDf = df[ind].copy()
  passKeyDf = passDf[keyCols].drop_duplicates()

  passKeyDf[passColName] = True
  keyDfLabeled = pd.merge(keyDfUnique, passKeyDf, on=keyCols, how='left')
  keyDfLabeled = keyDfLabeled.fillna(False)

  dfLabeled = pd.merge(df, keyDfLabeled, on=keyCols, how='left')

  return {'dfLabeled':dfLabeled, 'keyDfLabeled':keyDfLabeled}

'''
df = pd.DataFrame({'user_id':[1, 1, 2, 2, 3, 3, 4, 4],
                   'device':['pixel', 'sams', 'lg', 'lg', 'sams', 'pixel', 'nex', 'pixel'],
                   'country':['us', 'us', 'jp', 'jp', 'kr', 'kr', 'in', 'in']})

outDict = PartDfByKeysWrtCond(
  df=df, keyCols=['user_id'], condDict={'device':['pixel'],
  'country':['us', 'in']}, passColName='passesCond')

Mark(df)
Mark(outDict['dfLabeled'])
Mark(outDict['keyDfLabeled'])
'''

## create good pandas boxplots
def PandasBoxPlot(df, column, by, ylim=None, yscale=None, figSize=None):

  bp = df.boxplot(column=column, by=by)

  if yscale is not None:
    plt.yscale(yscale)
  [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
  fig = np.asarray(bp).reshape(-1)[0].get_figure()

  if figSize is not None:
    fig.set_size_inches(figSize[0], figSize[1])

  plt.xticks(rotation=90)
  axes = plt.gca()

  if ylim is not None:
    axes.set_ylim(ylim)

  plt.show()
