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

"""generic functions to calculate engagement metrics"""

## Calculate Dau for a dataframe with users and dates
def CalcDau(
  df, pltTitle='', pltIt=False, userCol='user_id', dateCol='date',
  pltLabel='', pltCol='b', propCoef=1.0):

  df = df[[userCol, dateCol]]
  def f(x):
    return(len(set(x)))
  g = df.groupby([dateCol])
  df2 = g.aggregate(f)
  dau = df2.reset_index()

  if pltIt:
    plt.plot(
        dau[dateCol], propCoef*dau[userCol],
        label=pltLabel, color=pltCol, alpha=0.8)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90, fontsize=10)
    plt.title(pltTitle, fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

  dau.columns = ['date', 'dau']
  dau['dau'] = propCoef * dau['dau']

  return dau

## calculate the number of events
def CalcEvents(df, valueCol, pltTitle='', pltIt=False, dateCol='date'):

  df = df[[dateCol, valueCol]]
  g = df.groupby([dateCol])
  df2 = g.aggregate(sum)
  dau = df2.reset_index()
  if pltIt:
    plt.plot(dau[dateCol], 10*dau[valueCol])
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90, fontsize=10)
    plt.title(pltTitle, fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

  return dau


## we consider the following temporal pattern:
##  .... [inactivityPeriod] .. [prePeriod] .... [PostPeriod]
## calculates retention in postPeriod given the user is present in prePeriod
## it also insures the user has not been active during inactivePeriod
# which is a proxy for new user

def CalcDailyRetention(
  df, inactiveDf, prePeriod, inactivePeriod=None,
  inactiveLength=None, pltIt=False):

  prePeriod = [FloatOrStr_toDatetime(x) for x in prePeriod]

  if inactivePeriod != None:
    inactivePeriod = [FloatOrStr_toDatetime(x) for x in inactivePeriod]
  else:
    base = prePeriod[0] -  datetime.timedelta(days=1)
    inactivePeriod = [base - datetime.timedelta(days=x) for x in range(0, inactiveLength)]

  df0 = inactiveDf[inactiveDf['date'].isin(inactivePeriod)]
  dfPre = df[df['date'].isin(prePeriod)]
  users = list(set(dfPre['user_id']) - set(df0['user_id']))
  df = df[df['user_id'].isin(users)]
  #print(df)
  dau = CalcDau(df, pltTitle='1dau for usage of either', pltIt=False)
  out = dau[dau['date'] >= max(prePeriod)]
  if pltIt:
    plt.plot(out['date'], out['user_id'])
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60, fontsize=10)
    plt.title('daily retention', fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

  return out
  #plt.legend(loc='upper left')

# it generates a date grid using two dates given in interval
def DateIntervalToGrid(interval):

  dtGrid = []
  if len(interval) == 2:
        start = interval[0]
        end = interval[1]
        step = datetime.timedelta(days=1)
        while start <= end:
          dtGrid.append(start)
          start += step

  return dtGrid

'''
postPeriod = ['20161023', '20161028']
postPeriod =  map(FloatOrStr_toDatetime, postPeriod)
print(DateIntervalToGrid(postPeriod))
'''

def CalcRetention(
    df, inactiveDf, inactivePeriod, prePeriod,
    postPeriod, interval=True,
    userCol='user_id', dateCol='date', pltIt=False):

  inactivePeriod = [FloatOrStr_toDatetime(x) for x in inactivePeriod]
  prePeriod = [FloatOrStr_toDatetime(x) for x in prePeriod]
  postPeriod = [FloatOrStr_toDatetime(x) for x in postPeriod]

  if interval:
    if len(inactivePeriod) == 2:
      inactivePeriod = DateIntervalToGrid(inactivePeriod)
    if len(prePeriod) == 2:
      prePeriod =  DateIntervalToGrid(prePeriod)
    if len(postPeriod) == 2:
      postPeriod =  DateIntervalToGrid(postPeriod)
  df0 = inactiveDf[inactiveDf[dateCol].isin(inactivePeriod)]
  dfPre = df[df[dateCol].isin(prePeriod)]
  dfPost = df[df[dateCol].isin(postPeriod)]
  users = list(set(dfPre[userCol]) - set(df0[userCol]))

  usersRetained = list(
      (set(dfPre[userCol]) & set(dfPost[userCol]))
      - set(df0[userCol]))

  out = len(usersRetained) / float(len(users))

  return out

## Calculate retention using other input
## lastDate is the day for which we are calculating retention. This is the last
## day of the retention period.
## postPeriodLength: the number of days for the post period which is the period
## we want to see if the user is retained
##  prePeriodLength (typical 1) is the length of the interval for which
## the user is appeared (for the first time after the inactivity period)
## these are the users for which we want to see if they are retained in post period
## inactiveLength: is the length for which we check inactivity before the prePeriod
## prePostGapLength is the gap between the prePeriods last day and PostPeriod first day
## If the inactivePeriodLength is None the we use dataSlice.date_range[1] as
## inactivePeriod[0] (beginning of the inactive period)
def CalcRetentionDate(
    df, inactiveDf, lastDate,
    postPeriodLength=7, prePeriodLength=1, inactiveLength=None,
    prePostGapLength=7, dateCol='date', userCol='user_id'):

  ## calculate the beginning of the postPeriod by subtracting the postPeriod length
  postPeriod1 = (
      datetime.datetime.strptime(lastDate, '%Y%m%d')
      - datetime.timedelta(days=postPeriodLength-1))

  postPeriod = [postPeriod1.strftime("%Y%m%d"), lastDate]
  # calculate the prePeriod
  # subtract the gap to get the last day of the prePeriod
  prePeriod2 = postPeriod1 -  datetime.timedelta(days=prePostGapLength)
  # subtract also the length of the preperiod
  # to get the beginning of the preperiod
  prePeriod1 = (
      postPeriod1
      - datetime.timedelta(days=(prePostGapLength + prePeriodLength-1)))
  prePeriod = [prePeriod1.strftime("%Y%m%d"), prePeriod2.strftime("%Y%m%d")]
  # calculate the inactivePeriod
  inactivePeriod2 = prePeriod1 -  datetime.timedelta(days=1)
  # if inactivePeriodLength is not given then we use the first day data
  # is available in the slice
  if (inactiveLength==None):
    inactivePeriod1 = min(inactiveDf[dateCol])  -  datetime.timedelta(days=2)
    inactivePeriod1 = inactivePeriod1.strftime("%Y%m%d")
  else:
    inactivePeriod1 = prePeriod1 -  datetime.timedelta(days=(1 + inactiveLength))
    inactivePeriod1 = inactivePeriod1.strftime("%Y%m%d")
  inactivePeriod = [inactivePeriod1, inactivePeriod2.strftime("%Y%m%d")]

  out = CalcRetention(
      df=df, inactiveDf=inactiveDf, prePeriod=prePeriod, postPeriod=postPeriod,
      inactivePeriod=inactivePeriod, interval=True, userCol=userCol, dateCol=dateCol)

  return out

'''
df = ...
prePeriod = ['20161021']
postPeriod = ['20161023', '20161025']
inactivePeriod = ['20161017', '20161018', '20161019', '20161020']
inactiveDf = df
out = CalcDailyRetention(
    df=df[['user_id', 'date']], inactiveDf=df,
    inactivePeriod=inactivePeriod, prePeriod=prePeriod)
CalcRetention(df, inactiveDf, inactivePeriod, prePeriod, postPeriod)
CalcRetentionDate(df=df, inactiveDf=inactiveDf, lastDate='20161104')
'''

# calculate retention for each date (last day), week 1 to week 2
def CalcRetentionOverTime(
    df, inactiveDf, start_date='20161020', end_date='20161105'):

  start = datetime.datetime.strptime(start_date, '%Y%m%d')
  end = datetime.datetime.strptime(end_date, '%Y%m%d')
  step = datetime.timedelta(days=1)
  dateGrid = []
  dtGrid = []
  while start <= end:
      dateGrid.append(start.strftime('%Y%m%d'))
      dtGrid.append(start)
      start += step
  outList = []
  for i in range(len(dateGrid)):
    lastDate = dateGrid[i]
    out = CalcRetentionDate(df=df, inactiveDf=inactiveDf, lastDate=lastDate)
    outList.append(out)
  df = pd.DataFrame({'date': dtGrid, 'value': outList})

  return df

# Calculate EngIn7 and metrics like that,
# which aggregate from a period up to the "date"
# The aggregation is done with respect to wrtCols
# which can be [user_id, feature] as an example
# The length of that period is "dayNum".
# Therefore the whole period is [date-dayNum+1, date] which includes date
def AggWrtPeriod(date, df, dateCol, wrtCols, dayNum, aggFcn, valueCol=None):

  if valueCol == None:
    valueCol = 'aggCol'
    df[valueCol] = df[dateCol].astype(str)
  ## sub-setting the data to the dates which qualify
  date2 = datetime.datetime.strptime(date, '%Y%m%d')
  date1 = date2 - datetime.timedelta(days=(dayNum - 1))
  df2 = df[(df[dateCol] >= date1) * (df[dateCol] <= date2)]
  df3 = df2[wrtCols + [valueCol]]
  g = df3.groupby(wrtCols)
  out = g.aggregate(aggFcn)
  out = out.reset_index()

  return out

# calculating user engagement
# this function will first aggregate with respect to wrtCols e.g.
# [userCol, featureCol] using aggFcn
# then it aggregates with respect to userCol using userAggFcn
# it also allows for a user sample period
# which is different from the aggregation period
# it also allows for the user sample to be picked from a different data frame
# for other wrtCols we provide a version of output
# which insures all possible combinations are present (cross join)
# userSamplingMethod='preceding' will take the users who appear
# on the previous period of same length
# userSamplingMethod == 'same' will take the users who appear in same period,
# but note that more wrtCols values might be picked up
def CalcEngage(
    date, df, dateCol, wrtCols, userCol, dayNum, aggFcn,
    valueCol=None, sampledUsers=None,
    userSamplingDf=None, userSamplingDates=None,
    userSamplingMethod=None, userAggFcn=np.mean):
  # first we aggregate wrt wrtCols
  aggDf = AggWrtPeriod(
      date=date, df=df, dateCol=dateCol, wrtCols=wrtCols,
      dayNum=dayNum, aggFcn=aggFcn, valueCol=valueCol)

  if userSamplingDf == None:
    userSamplingDf = df
  if userSamplingMethod == 'preceding':
    date2 = (
        datetime.datetime.strptime(date, '%Y%m%d')
        - datetime.timedelta(days=dayNum))
    date1 = (
        datetime.datetime.strptime(date, '%Y%m%d')
        - datetime.timedelta(days=(dayNum*2-1)))
    userSamplingDates = [date1.strftime('%Y%m%d'), date2.strftime('%Y%m%d')]
  print('user sampling dates')
  print(userSamplingDates)
  # if userSamplingMethod == 'same':
  # date2 = datetime.datetime.strptime(date, '%Y%m%d')
  # date1 = date2 - datetime.timedelta(days=(dayNum-1))
  # userSamplingDates = [date1.strftime('%Y%m%d'), date2.strftime('%Y%m%d')]

  # if sampledUsers is directly given then we take it.
  # if not we use the default value for sampled users from df:
  if (sampledUsers == None):
    sampledUsers = list(set(aggDf[userCol]))

  if (userSamplingDates != None):
    date1 = userSamplingDates[0]
    date2 = userSamplingDates[1]
    date1 = datetime.datetime.strptime(date1, '%Y%m%d')
    date2 = datetime.datetime.strptime(date2, '%Y%m%d')
    userSamplingDf2 = (
        userSamplingDf[
            (userSamplingDf[dateCol] >= date1)
            * (userSamplingDf[dateCol] <= date2)])
    sampledUsers = list(set(userSamplingDf2[userCol]))

  ## cross joining sampled users with other fields if available e.g. features
  fullDict = {userCol: sampledUsers}
  fullDf = pd.DataFrame(fullDict)
  fullDf['tempKey'] = 1
  wrtCols2 = list(set(wrtCols) - set([userCol]))
  for i in range(len(wrtCols2)):
    col = wrtCols2[i]
    x = list(set(list(aggDf[col]) + list(userSamplingDf[col])))
    fullDict[col] = x
    tempDf = pd.DataFrame({col: x})
    tempDf['tempKey'] = 1
    fullDf = pd.merge(fullDf, tempDf, on=['tempKey'])
  del fullDf['tempKey']
  fullDf['aggCol2'] = 0
  aggFullDf = pd.merge(fullDf, aggDf, how='left', on=wrtCols)
  aggFullDf = aggFullDf.fillna(0.0)
  aggFullDf['aggCol'] = aggFullDf['aggCol'] + aggFullDf['aggCol2']
  del aggFullDf['aggCol2']

  aggDf2 = aggDf.copy()
  aggFullDf2 = aggFullDf.copy()

  #print(aggFullDf2)
  aggDf2['tempKey'] = 1
  aggFullDf2['tempKey'] = 1
  del aggDf2[userCol]
  del aggFullDf2[userCol]
  cols = wrtCols2 + ['tempKey']
  g1 = aggDf2.groupby(cols)
  g2 = aggFullDf2.groupby(cols)
  #print(aggFullDf2)
  #integrating user out:
  aggUserDf = g1.aggregate(userAggFcn)
  aggUserFullDf = g2.aggregate(userAggFcn)
  aggUserDf = aggUserDf.reset_index()
  aggUserFullDf = aggUserFullDf.reset_index()
  del aggUserDf['tempKey']
  del aggUserFullDf['tempKey']

  outDict = {
    'aggDf':aggDf, 'aggFullDf':aggFullDf, 'aggUserDf':aggUserDf,
    'aggUserFullDf':aggUserFullDf}

  return outDict
