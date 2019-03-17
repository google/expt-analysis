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

"""
Some simple functions to facilitate predictive modeling in python
"""

def SecToDateTime(x, threshold=datetime.datetime(2014, 1, 1, 1)):

  if x == None or math.isnan(x):
    y = None
  else:
    y = datetime.datetime.fromtimestamp(x)
    if y < threshold:
      y = None

  return y


def FindBetweenString(start, end, s):

  import re
  result = re.search(start + '(.*)' + end, s)

  return result.group(1)


#print calendar.isleap(1900)
## maps a datetime to a continuous time value
def DateTime_toConti(t):

  import calendar
  year = t.year
  l = 365 + calendar.isleap(year)
  conti = (t.year + t.dayofyear/float(l) + t.hour/(l*24.0) +
    t.minute/(l*24.0*60.0) + t.second/(l*24.0*3600.0))

  return conti

### we can find IQRs per item (query,result)
### but we find an IQR for all
def IqrFcn(p2=0.8, p1=0.2):

  def F(x):
    return x.quantile(p2) - x.quantile(p1)

  return F

def Lquantile(x, p):
  ## see definition by Hosseini thesis, UBC, 2010, page 124
  if type(p) == float:
    p = [p]
  x = pd.Series(x)
  y = np.sort(x)
  n = len(x)
  ## below is simply the vector n*p in R
  m = p*(np.array([n]*len(p)))
  #print m
  m1 = map(math.ceil, m)
  m2 = map(int, m1)
  m3 = map(lambda x: max([x-1, 0]), m2)
  y = list(y)
  z = [y[i] for i in m3]
  z = np.array(z)

  if len(z) == 1:
    z = z[0]

  return z

def Rquantile(x, p):
  ## see definition by Hosseini thesis, UBC, 2010, page 124
  if type(p) == float:
    p = [p]
  x = pd.Series(x)
  y = np.sort(x)
  n = len(x)
  ## below is simply the vector n*p in R
  m = p*(np.array([n]*len(p)))
  #print m
  m1 = map(math.floor,m)
  m2 = map(int,m1)
  m3 = map(lambda x: min([x,n-1]),m2)
  y = list(y)
  z = [y[i] for i in m3]
  z = np.array(z)
  if len(z) == 1:
    z = z[0]

  return z
'''
# example
x = [0, 1]
p = [1, 1, 0, 0.1, 0.2, 0.9, 1]
print Lquantile(x, p)
x = [0, 1, 2]
print Lquantile(x,p)
p = [0, 0.1, 0.2, 0.5, 0.9, 0.99, 1]
x = range(10)
print Lquantile(x, p)
'''
def TrimmedMeanFcn(p2=0.8, p1=0.2):

  def F(x):
    x2 = Lquantile(x, p2)
    x1 = Rquantile(x, p1)
    z = [i for i, i in enumerate(x) if (i<=x2 and i>=x1)]
    z = pd.Series(z)
    return z.mean()

  return F

TrimmedMean = TrimmedMeanFcn(p2=0.75, p1=0.25)

'''
# example
x = pd.Series(range(10))
print(TrimmedMean(x))
'''
def Quartile3(x):
  x = pd.Series(x)
  return x.quantile(0.75)

def Quartile1(x):
  x = pd.Series(x)
  return x.quantile(0.25)

## numpy median created some NULL so screw that. we redefine median here.
def Quartile2(x):
  x = pd.Series(x)
  return x.quantile(0.5)

## calculate auto-correlation
def AutoC(x):
  """
  http://en.wikipedia.org/wiki/AutoCorrelation#Estimation
  """
  x = x[~np.isnan(x)]
  if len(x) < 2:
    return 0
  n = len(x)
  variance = np.var(x)
  x = x - np.mean(x)
  r = np.correlate(x, x, mode = 'full')[-n:]
  # assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
  result = r/(variance*(np.arange(n, 0, -1)))

  return result[1]


## skewness: second skewness of pearson
def Pskew2(x):

  if float(np.std(x)) == 0:
      return 0.0

  return 3.0*float(np.mean(x)-np.median(x))/float(np.std(x))

##  change string timestamp to datetime in python
# this can handle with or without seconds
# with map! it takes 50 percent of for loop
def Build_datetimeStr(df, timeCol='timestamp'):

 def Build_datetimeStr_atom(time1):
    time1 = time1.replace('/',' ')
    time1 = time1.replace('-',' ')
    if time1.count(':') == 2:
        time1 = datetime.datetime.strptime(time1, '%d %m %Y %H:%M:%S')
    elif time1.count(':') == 1:
        time1 = datetime.datetime.strptime(time1, '%d %m %Y %H:%M')
    return time1

	n = len(df)
	dt = map(Build_datetimeStr_atom, df[timeCol])

	return dt

## segment the data with respect to a variable such as week
## the segment variable: groupCol
## the within variable order: elementCol
## we fill NA in place of missing elements
def DfGroup_elementFill(df, valCol='value', groupColName='group',
    elementColName='element', naFill=None):

  groupCol = df[groupColName]
  elementCol = df[elementColName]
  groupList = list(set(groupCol))
  elementList = list(set(elementCol))
  dfG = pd.DataFrame({'key':[1]*len(groupList), groupColName:groupList})
  dfE = pd.DataFrame({'key':[1]*len(elementList), elementColName:elementList})
  df1 = df[[groupColName, elementColName, valCol]]
  df2 = pd.merge(dfG, dfE,on='key')
  df2 = df2[[groupColName, elementColName]]
  df3 = pd.merge(df2, df1, how='left', on=[groupColName, elementColName])

  return df3

def WeekDayToStr(x):

  d = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}

  return d[x]

## this function gets a datetime vector and creates new temporal columns
# eg year, week of year, etc
def TimeSeries_AddTimeInfo(dt):

  import datetime
  n = len(dt)
  year = []
  month = []
  day = []
  yearMonth = []
  yearWeek = []
  monthDay = []
  yearWeekDay = []
  weekDay = []
  hour = []
  minute = []
  second = []
  wdayHrMinSec = []
  conti_wdayTime = []
  conti_tod = []
  doy = []
  woy = []
  conti_year = []
  date = []
  str_weekDay = []
  isWeekend = []

  for i in range(n):

    dt0 = dt[i]
    date.append(str(dt0.date()))
    tt = dt0.timetuple()
    day.append(dt0.day)
    doy.append(tt.tm_yday)
    year.append(dt0.year)
    yearLength = datetime.datetime(dt0.year, 12, 31).timetuple().tm_yday
    conti_year1 = (dt0.year +
      (tt.tm_yday-1 + dt0.hour/24 + dt0.minute/(24*60) +
      dt0.second/(24*60*60))/float(yearLength))
    conti_year.append(conti_year1)
    month.append(dt0.month)
    yearMonth.append((dt0.year, dt0.month))
    monthDay.append((dt0.month, dt0.day))
    yearWeekDay.append(dt0.isocalendar())
    yearWeek.append(dt0.isocalendar()[0:2])
    woy.append(dt0.isocalendar()[1])
    weekDay.append(dt0.isoweekday())
    u = WeekDayToStr(dt0.isoweekday())
    str_weekDay.append(u)

    if dt0.isoweekday() > 4:
        isWeekend.append('Yes')
    else:
        isWeekend.append('No')
    hour.append(dt0.hour)
    minute.append(dt0.minute)
    second.append(dt0.second)
    wdayHrMinSec.append((dt0.isoweekday(), dt0.hour, dt0.minute, dt0.second))
    x = dt0.hour/24.0 + dt0.minute/(24.0*60.0) + dt0.second/(24.0*3600)
    conti_tod.append(x)
    y = (dt0.isoweekday()-1) + x
      conti_wdayTime.append(y)

  timeDict = {
      'date':date, 'year':year, 'month':month, 'yearMonth':yearMonth,
      'yearWeek':yearWeek, 'yearWeekDay':yearWeekDay, 'weekDay':weekDay,
      'hour':hour, 'minute': minute, 'second': second,
      'wdayHrMinSec':wdayHrMinSec,'conti_tod':conti_tod,
      'conti_wdayTime':conti_wdayTime,'doy':doy, 'conti_year':conti_year,
      'woy':woy,'monthDay':monthDay, 'isWeekend':isWeekend,
      'str_weekDay':str_weekDay}

  timeDict = pd.DataFrame(timeDict)

  return timeDict

### Add flag to temporal data
def TimeSeries_addFlag(dt, flagData, timeCol='date', flagCol='event',

  flagText=['Yes','No'], flagBinary='flagBinary', dateFormat='%Y-%m-%d'):
  timeDf = TimeSeries_AddTimeInfo(dt)
  flagData = flagData[[timeCol, flagCol]]
  flagData[flagBinary] = flagText[0]
  df = pd.merge(timeDf, flagData, on=timeCol, how='left')
  df[flagCol] = df[flagCol].fillna('Regular')
  df[flagBinary] = df[flagBinary].fillna(flagText[1])

  return df

### rounding up a timestamp up to delta of 5 minute
def FloorDatetime_xmin(time1, delta=5):

  minute = int(round(time1.minute/delta)*delta)
  time2 = datetime.datetime(time1.year, time1.month,
                            time1.day, time1.hour, minute)
  return time2

### shift a timestamp in seconds
def ShiftDatetime(dt, seconds):

  tss = time.mktime(dt.timetuple())
  tss2 = tss + seconds
  #dt2 =  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tss2))
  dt2 = datetime.datetime.fromtimestamp(tss2)

  return dt2

## build a lag dictionary for a variable
def BuildLagDf(y, r, colName=''):
  # y is the variable
  # r is the lag number
  dict1 = {}
  y = list(y)
  n = len(y)
  z = list(y)
  for i in range(r):
    k = i + 1
    z.insert(0, None)
    del z[(n)]
    string = colName + '_lag' + str(k)
    dict1[string] = list(z)
  df = DataFrame(dict1)

  return df

def ModR2(yPred, yValid, mean=None, yTrain=[], p=2):

  if len(yTrain) > 0 and mean == None:
      mean = yTrain.mean()
  elif mean == None:
      mean = yValid.mean()
  E = (np.array(yPred)-np.array(yValid))
  E = pd.Series(E)
  Eclean = E[~E.isnull()]

  if len(Eclean) < len(E):
    print('Warning: some Nans in the pred')
    print('length of data:' + str(len(yValid)))
    print('length of Error vector after removing Nans:' + str(len(Eclean)))
  SSres = sum(abs(Eclean)**p)
  SStot = sum(abs(yValid-mean)**p)

  R2 = None
  if SStot > 0:
      R2 = 1 - (SSres/SStot)

  return R2

def AssessContiPred(yPred, yValid):

  R2 = ModR2(yPred=yPred, yValid=yValid, mean=None, yTrain=[])
  E = np.array(yPred)-np.array(yValid)
  E = pd.Series(E)
  Eclean = E[~E.isnull()]

  if len(Eclean) < len(E):
    print('Warning: some Nans in the pred')
    print('length of data:' + str(len(yValid)))
    print('length of Error vector after removing Nans:' + str(len(Eclean)))

  AE =  abs(E)
  ME = E.mean()
  E25 = pd.Series(E).quantile(0.25)
  E50 = pd.Series(E).quantile(0.50)
  E75 = pd.Series(E).quantile(0.75)
  MAE = AE.mean()
  SE = E**2
  MSE = SE.mean()
  VAE = AE.var()
  VSE = SE.var()
  AE95 = pd.Series(AE).quantile(0.95)
  AE05 = pd.Series(AE).quantile(0.05)
  AE25 = pd.Series(AE).quantile(0.25)
  AE75 = pd.Series(AE).quantile(0.75)
  AE50 = pd.Series(AE).quantile(0.5)
  RMSE = np.sqrt(MSE)
  outDict = {'R2':R2, 'MAE':MAE, 'MSE':MSE, 'RMSE':RMSE, 'VAE':VAE, 'VSE':VSE,
             'AE95':AE95, 'AE05':AE05, 'AE50':AE50, 'AE75':AE75, 'AE25':AE25,
             'E25':E25, 'E75':E75, 'E50':E50}

  return outDict

def ApplyFcnRow(explan, F):
    return explan.apply(F, axis=1)

'''
# example:
n = 100
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
x3 = np.random.normal(size=n)
x4 = np.random.normal(size=n)
y = (x1+x2+x3)/4.0
newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3,'x4':x4}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3','x4']]
explan = df2[['x1','x2','x3','x4']]

fcn = np.median
yPred = predict(explan,fcn)
plt.scatter(y,yPred)
AssessContiPred(yPred=yPred,yValid=y)

fcn = np.mean
yPred = predict(explan,fcn)
plt.scatter(y,yPred)
AssessContiPred(yPred=yPred,yValid=y)
'''

'''
# random forest example
from sklearn.ensemble import RandomForestClassifier
n = 100
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
x3 = np.random.normal(size=n)
x4 = np.random.normal(size=n)
y = (x1+x2+x3)/4.0
newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3,'x4':x4}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3','x4']]
explan = df2[['x1','x2','x3','x4']]
mat = explan.as_matrix()
rf = RandomForestRegressor()
rf.fit(explan,y)
Yfit = rf.predict(explan)
plt.scatter(out,y)
'''
def FitAggModel(data, explanCols, yCols, aggFcn=np.mean):

  print('** agg model was run')
  df2 = data.copy()
  dfAgg = df2.groupby(explanCols)[[yCols]].agg(lambda x: aggFcn(x))
  dfAgg = dfAgg.reset_index()
  df2['uncert'] = 1
  dfUncert = df2.groupby(explanCols)[['uncert']].agg(lambda x: np.sum(x))
  dfUncert = dfUncert.reset_index()

  def predFcn(explanDf):
    mergedDf = pd.merge(explanDf, dfAgg, how='left', on=explanCols)
    yPred = mergedDf[yCols]
    return yPred
  def uncertFcn(explanDf):
    mergedDf = pd.merge(explanDf, dfUncert, how='left', on=explanCols)
    uncert = mergedDf['uncert']
    return(uncert)

  explanDf = data[explanCols]
  yFit = predFcn(explanDf)
  resDict = {'mod':dfAgg, 'yFit':yFit, 'predFcn':predFcn,
             'uncertFcn':uncertFcn, 'uncert':dfUncert}

  return resDict


'''
n = 100
x1 = np.round(np.random.normal(size=n))
x2 = np.round(np.random.normal(size=n))
x3 = np.round(np.random.normal(size=n))
x4 = np.round(np.random.normal(size=n))
y = (x1+x2+x3)/4.0
newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3,'x4':x4}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3','x4']]
explanCols = ['x1','x2','x3']
explanDf = df2[explanCols]
res = FitAggModel(data=df2,explanCols=explanCols,yCols='y',aggFcn=np.mean)
yFit = res['yFit']
predFcn = res['predFcn']
yFit2 = predFcn(explanDf)
plt.scatter(y,yFit2)
'''


'''
n = 50
x1 = np.round(np.random.normal(size=n))
x2 = np.round(np.random.normal(size=n))
x3 = np.round(np.random.normal(size=n))
x1 = np.array(map(math.floor,x1))
x2 = np.array(map(math.floor,x2))
x3 = np.array(map(math.floor,x3))
print x1
y = (x1+x2+x3)/4.0
newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3']]
print df2
explanCols = ['x1','x2','x3']
explanDf = df2[explanCols]
res = FitAggModel(data=df2,explanCols=explanCols,yCols='y',aggFcn=np.mean)
print res['uncert']
yFit = res['yFit']
predFcn = res['predFcn']
uncertFcn = res['uncertFcn']
yFit2 = predFcn(explanDf)
uncertFit = uncertFcn(explanDf)
plt.figure(1)
plt.scatter(y,yFit2)
plt.figure(2)
plt.scatter(y,uncertFit)
'''


### fit linear model to continuous (and categ?) y and report results
def fitPredModel(data, modExpr=None, explanCols=None, yCols=None,
                 model='linear', fullModel=False, indTrain=None,
                 indValid=None, predFcn=None, aggFcn=np.mean,
                 printSummary=False, pltIt=False):

  import statsmodels.api as sm
  from patsy import dmatrices
  from scipy.stats.stats import pearsonr

  def df_series(a):
    a = a[a.columns[0]]
    return a

  data = data.dropna(subset=data.columns)

  def buildExplanY(data, explanCols=None, yCols=None):
    if not modExpr == None:
        y, X = dmatrices(modExpr, data=data, return_type='dataframe')
        explanCols = list(X.columns)
        yCols = FindBetweenString(start='', end='~', s=modExpr)
        y = df_series(y)
    else:
        print(yCols)
        y = data[yCols]
        X = data[explanCols]
    return {'X':X, 'y':y, 'explanCols':explanCols, 'yCols':yCols}

  explanY = buildExplanY(data, explanCols=explanCols, yCols=yCols)
  y = explanY['y']
  X = explanY['X']
  explanCols = explanY['explanCols']
  yCols = explanY['yCols']

  def explanFcn(newData):
    out = buildExplanY(newData)
    X = out['X']
    if model == 'linear':
        X = X.as_matrix()
    return X

  n = len(data)

  if fullModel:
    indTrain = range(n)
    indValid = range(n)

  if indTrain == None and not fullModel:
    import random
    k = round(n/2)
    k = int(k)
    indTrain = random.sample(range(n), k)
    indTrain = np.sort(indTrain)

  if indValid == None and not fullModel:
    indValid = list(set(range(n)) - set(indTrain))

  yTrain = y.iloc[indTrain]
  yValid = y.iloc[indValid]
  Xtrain = X.iloc[indTrain]
  Xvalid = X.iloc[indValid]
  modSummary = None
  resMod = None
  uncertFcn = None
  uncert = None

  if model == 'linear':
    mod = sm.OLS(yTrain, Xtrain)
    resMod = mod.fit()
    modPredFcn = resMod.predict
    modSummary = resMod.summary()
    if (printSummary):
      print(resMod.summary())
      print('*******')

  if model == 'rf':
    from sklearn.ensemble import RandomForestRegressor
    Xtrain = Xtrain.as_matrix()
    Xvalid = Xvalid.as_matrix()
    yTrain = np.array(yTrain)
    yValid = np.array(yValid)
    rf = RandomForestRegressor()
    resMod = rf.fit(Xtrain,yTrain)
    modPredFcn = resMod.predict
    yTrain = pd.Series(yTrain)
    yValid = pd.Series(yValid)

  if model == 'agg':
    data2 = X
    data2[yCols] = y
    res = FitAggModel(data2, explanCols, yCols, aggFcn)
    modPredFcn = res['predFcn']
    resMod = res['mod']
    uncertFcn = res['uncertFcn']
    uncert = res['uncert']

  if predFcn is not None:
    def modPredFcn(x):
      return ApplyFcnRow(x, fcn=predFcn)
    resMod = None

  yFit = modPredFcn(Xtrain)
  yPred = modPredFcn(Xvalid)
  yFit = pd.Series(yFit)
  yPred = pd.Series(yPred)

  if pltIt:
    plt.scatter(yValid, yPred, color='red', alpha=0.05)
    plt.title('validation set')

  corr = str(pearsonr(yPred, yValid))
  Efit = (np.array(yTrain)-np.array(yFit))
  Efit = pd.Series(Efit)
  Efit= Efit[~Efit.isnull()]

  if len(Efit) < len(yTrain):
    print('Warning: some Nans in the fit')
    print('length of data:' + str(len(yTrain)))
    print('length of Error vector after removing Nans:' + str(len(Efit)))

  Epred = (np.array(yValid)-np.array(yPred))
  Epred = pd.Series(Epred)
  Epred = Epred[~Epred.isnull()]

  if len(Epred) < len(yValid):
    print('Warning: some Nans in the pred')
    print('length of data:' + str(len(yValid)))
    print('length of Error vector after removing Nans:' + str(len(Epred)))


  fitR2 = ModR2(yPred=yFit, yValid=yTrain, mean=yTrain.mean(), yTrain=yTrain)
  fitMAE =  abs(Efit).mean()
  fitRMSE =  math.sqrt((abs(Efit)**2).mean())
  predR2 = ModR2(yPred,yValid, mean=yTrain.mean(), yTrain=yTrain);
  predMAE = abs(Epred).mean()
  predRMSE = math.sqrt((abs(Epred)**2).mean())

  return {'mod':resMod, 'modPredFcn':modPredFcn,
    'summary':modSummary, 'yFit':yFit, 'yTrain':yTrain, 'yValid':yValid,
    'uncertFcn':uncertFcn, 'uncert':uncert,
    'yPred':yPred, 'Xtrain':Xtrain, 'Xvalid':Xvalid, 'explanFcn':explanFcn,
    'fitR2':fitR2, 'predR2':predR2, 'predMAE':predMAE, 'predRMSE':predRMSE,
    'fitMAE':fitMAE, 'fitRMSE':fitRMSE}

'''
# example:
n = 100
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
x3 = np.random.normal(size=n)
x4 = np.random.normal(size=n)
y = 10+0*x1+2*x2+0*x3+x4

newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3,'x4':x4}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3','x4']]

plt.figure(1)
plt.scatter(df2['x2'],df2['y'])


explanString = 'x1+x2+x3+x4'
yString = 'y'
modExpr = yString + '~' + explanString

plt.figure(2)
fitPredModel(modExpr,data=df2)
'''

'''
n = 100
x1 = np.round(np.random.normal(size=n))
x2 = np.round(np.random.normal(size=n))
x3 = np.round(np.random.normal(size=n))
x4 = np.round(np.random.normal(size=n))
y = (x1+x2+x3)/4.0 + np.round(np.random.normal(size=n))
newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3,'x4':x4}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3','x4']]
explanCols = ['x1','x2','x3']
explanDf = df2[explanCols]

res = fitPredModel(data=df2,modExpr=None,explanCols=explanCols,yCols=yCols,
    model='linear',fullModel=False,
    indTrain=None,indValid=None,predFcn=None,printSummary=False,pltIt=False)

res = FitAggModel(data=df2,explanCols=explanCols,yCols='y',aggFcn=np.mean)
yFit = res['yFit']
predFcn = res['predFcn']
yFit2 = predFcn(explanDf)
plt.scatter(y,yFit2)
'''

'''
n = 100
x1 = np.round(np.random.normal(size=n))
x2 = np.round(np.random.normal(size=n))
x3 = np.round(np.random.normal(size=n))
x4 = np.round(np.random.normal(size=n))
y = (x1+x2+x3)/4.0 + np.round(np.random.normal(size=n))
newDict = {'y':y,'x1':x1,'x2':x2,'x3':x3,'x4':x4}
df2 = pd.DataFrame(newDict)[['y','x1','x2','x3','x4']]
explanCols = ['x1','x2','x3']
explanDf = df2[explanCols]


#print df2
print explanCols
print yCols

## linear model with column names
res = fitPredModel(data=df2,modExpr=None,explanCols=explanCols,yCols=yCols,
    model='linear',fullModel=False,
    indTrain=None,indValid=None,predFcn=None,printSummary=False,pltIt=False)

## linear model with model expression
modExpr = 'y~x1+x2+x3'
res = fitPredModel(data=df2,modExpr=modExpr,explanCols=None,yCols=None,
    model='linear',fullModel=False,
    indTrain=None,indValid=None,predFcn=None,printSummary=False,pltIt=False)
## agg model with Full Model
res = fitPredModel(data=df2,modExpr=None,explanCols=explanCols,yCols=yCols,
    model='agg',fullModel=True,
    indTrain=None,indValid=None,predFcn=None,printSummary=False,pltIt=True)

'''

### Fourier series matrix
def Fseries(data, colName, omega=2*math.pi, order=1):

  df = {}
  x = data[colName]
  x = np.array(x)
  columns = []

  for i in range(order):
    k = i + 1
    sincoln = 'sin' + str(k) + '_' + colName
    coscoln = 'cos' + str(k) + '_' + colName
    columns.append(sincoln)
    columns.append(coscoln)
    u = omega*k*x
    sin = map(math.sin, u)
    cos = map(math.cos, u)
    df[sincoln] = sin
    df[coscoln] = cos

  return {'df':DataFrame(df), 'columns':columns}

''' example
x = np.linspace(2.0, 3.0, num=100)
data1 = DataFrame({'x':x})
FS = Fseries(data=data1,colName='x',omega=2*math.pi,order=2)
'''

def AddFseries(data, colNames, omegas=None, orders=None):

  k = len(colNames)
  if omegas == None:
    omegas = [2*math.pi]*len(colNames)
  if orders == None:
    orders = [1]*len(colNames)
  outData = data
  outColumns = []

  for i in range(k):
    colName = colNames[i]
    omega = omegas[i]
    order = orders[i]
    FS = Fseries(data=data, colName=colName, omega=omega, order=order)
    FSdf = FS['df']
    FScolumns = FS['columns']
    outData = concat([outData, FSdf], axis=1)
    outColumns = outColumns + FScolumns

  return {'df':outData,'columns':outColumns}

'''
# example
x = np.linspace(2.0, 3.0, num=100)
y = np.linspace(3.0, 4.0, num=100)
data = DataFrame({'x':x, 'y':y})
res = AddFseries(data,colNames=['x','y'],
  omegas=[2*math.pi,4*math.pi],orders=[1,2])
data1 = res['df']
'''

## add interaction of variables to data
def AddInterac(data, s1, s2):

  n = len(s1)
  m = len(s2)

  for i in range(n):
    for j in range(m):
      k = s1[i]
      l = s2[j]
      name1 = data.columns[k]
      name2 = data.columns[l]
      name = name1 + '_' + name2;
      z = data.icol(k)*data.icol(l)
      data[name] = z

  return data

##
def InteracExpr(s1, s2, sep='+'):

  n = len(s1)
  m = len(s2)
  expr = ''
  for i in range(n):
    for j in range(m):
      name1 = s1[i]
      name2 = s2[j]
      name = name1 + '*' + name2;
      if (i+j)==0:
        expr = name
      else:
        expr = expr + sep + name

  return expr

## re-order columns of the data set via bootstrap
def BootsCol(data, ss=None, replace=True):

  k = len(data.columns)

  if ss is None:
    ss = k
  sample = np.random.choice(k, ss, replace=replace)
  l = data.columns[sample]
  data2 = data[l]

  return data2

#dfColSwapped=BootsCol(dfBu, ss=20, replace=False)


def ExpandDf_viaColSubset(data, subsetSize, copyNum):

  k = len(data.columns)
  # print k
  import itertools
  l = list(itertools.combinations(range(k), subsetSize))
  l2 = l
  l2.pop(0)
  import random
  random.shuffle(l2)
  print(l2)
  # print l
  num = len(l)
  fullset = set(range(k))
  outData = data.copy()
  colNames = data.columns
  part1 = range(subsetSize)
  part2 = fullset.difference(part1)
  part1 = list(part1)
  part2 = list(part2)
  range(1, copyNum)

  for i in range(1,copyNum):
    data2 = data.copy()
    ## i+1 because we already have the data
    subset = set(l2[i])
    subsetComp = fullset.difference(subset)
    list1 = list(subset)
    list2 = list(subsetComp)
    colNames1 = [colNames[i] for i in list1]
    data2.iloc[:, part1] = data.iloc[:, list1].values
    data2.iloc[:, part2] = data.iloc[:, list2].values
    data2.columns = data.columns
    outData = outData.append(data2)

  return outData

def ExpandDf_viaColReorder(data, copyNum, replace=True):

  outData = data.copy()

  for i in range(copyNum):
    data2 = BootsCol(data, ss=None, replace=replace)
    data2.columns = data.columns
    outData = outData.append(data2)

  return outData


## bootstrap the rows
def BootsRow(data, ss=None, replace=True):

  n = len(data)
  if ss == None:
    ss = n
  sample = np.random.choice(n, ss, replace=replace)
  data2 = data.iloc[sample,:]

  return data2

#dfRowSwapped = BootsRow(data=dfBu,replace=False)

## comparing measure stability, ss=k vs ss=k
def SplitDfCols(data, ind1=range(10), ind2=range(10,20)):

  colNames = data.columns
  cols1 = [colNames[x] for x in ind1]
  cols2 = [colNames[x] for x in ind2]
  data1 = data[cols1]
  data2 = data[cols2]

  return {'df1':data1, 'df2':data2}

## creates a large outlier or small
## if upper and lower are specified, uses them as outlier values
## if not it uses the min and max as well as classical definition to build upper/lower
## you can choose if you need upper or lower outliers or want to choose automatically
def CreateOutlier(x, upper=None, lower=None, side='free'):

  m = min(x)
  M = max(x)
  q1 = Lquantile(x, 0.25)
  q3 = Rquantile(x, 0.75)
  med = (Rquantile(x, 0.5) + Lquantile(x, 0.5))/2
  iqr = q3 - q1
  u = q3 + 1.5*iqr
  l = q1 - 1.5*iqr
  u = max([u,M])
  l = min([l,m])

  if upper is None:
  	upper = u
  if lower is None:
    lower = l
  if side == 'low':
    return lower
  if side == 'high':
    return upper
  if side == 'free':
    d1 = upper - q3
    d2 = q1 - lower
    if d1 <= d2:
        return lower
    else:
        return upper

## create outliers for a Df
def CreateOutlierDf(df, upper=None, lower=None, side='free'):

  def F(x):
    CreateOutlier(x, upper=upper, lower=lower, side=side)
  out = df.apply(F, axis=1)

  return out

## ordering the rows of a data frame
def OrderDfRows(df):

  n = len(df)
  for i in range(n):
    x = df.iloc[i, :].values
    x = np.array(x)
    y = np.sort(x)
    df.iloc[i, :] = list(y)

  return df

def FindClose(x, x0, epsilon):

  upper = x0 + epsilon
  lower = x0 - epsilon
  z = [u for u in x if u>lower and u<upper]

  return z

def FindNeighbNum(x, epsilon):

  n = len(x)
  out = []
  for i in range(n):
    x0 = x[i]
    z = FindClose(x, x0, epsilon)
    out.append(len(z))

  return out

def FindSemiModeInd(x, epsilon, agreeProp=0.5):

  num = FindNeighbNum(x,epsilon)
  mxInd = None
  mx = max(num)
  if float(mx)/float(len(x)) < agreeProp:
    return None
  maxInd = [i for i, j in enumerate(num) if j==mx]
  return maxInd

def FindSemiMode(x, epsilon, agreeProp=0.5):

  ind = FindSemiModeInd(x, epsilon, agreeProp)
  if ind == None:
    return None

  return [x[i] for i in ind]

## propThresh is the minimum needed agreement
def FindModeAvg(x, epsilon, method='hard', lift=2, agreeProp=0.5):
  semiMode = FindSemiMode(x, epsilon, agreeProp)

  if semiMode == None:
    return np.mean(x)
  upper = max(semiMode)
  lower = min(semiMode)
  u = [u for u in x if u >= lower and u <= upper]
  v = [w for w in x if w < lower or w > upper]
  out = np.mean(u)

  if method == 'soft' and (len(v) > 0):
    out = float(lift*np.sum(u) + np.sum(v))/float(lift*len(u) + len(v))

  return out

def FindModeAvgFcn(epsilon, method='hard', lift=2, agreeProp=0.5):

  def F(x):
      return FindModeAvg(x, epsilon, method=method, lift=lift)

  return F


'''
x=[2,2,2,3,3,3,3,1]
print FindSemiModeInd(x,epsilon=1)
print FindSemiMode(x,epsilon=1)
print FindModeAvg(x,epsilon=1)


x=[3,2.5,-10]
print FindSemiModeInd(x,epsilon=1)
print FindSemiMode(x,epsilon=1)
print FindModeAvg(x,epsilon=1)

x=[3,3,2.5,2.5,-10]
print FindSemiModeInd(x,epsilon=1)
print FindSemiMode(x,epsilon=1)
print FindModeAvg(x,epsilon=1)


x = [3,3,10,9.5,10.5]
print FindSemiModeInd(x,epsilon=1)
print FindSemiMode(x,epsilon=1)
print FindModeAvg(x,epsilon=1)
'''
