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

""" Functions to find first entry points in time-stamped data

This includes functions to find previous event/usage (entry point)
before given conditions hold: FindPrev
Also it finds the entry point to the next time the condition holds
This is useful if we want to compare the very first entry point to
the later entry point
"""

## finds previous event given in "usageCol"
# previous to the first time the conditions given by
# condCols, conDValues hold
# also it will only accepts an event as a previous event
# if its not off more than the time gap
# if the conditions do not hold or the gap is too big
# it will return None
def FindPrev0(df,
              condCols,
              condValues,
              usageCol='cond',
              userCol='user_id',
              timeCol='ts',
              timeGap=10 * 60):

  df =  df.reset_index(drop=True)
  df = df.sort_values([timeCol])
  df = df.reset_index(drop=True)

  condDict = {}
  for i in range(len(condCols)):
    col = condCols[i]
    if condValues[i] is not None:
      condDict[col] = [condValues[i]]

  dfCond = df.copy()
  if len(condDict) > 0:
    dfCond = SubDf_withCond(df=df, condDict=condDict, resetIndex=False)

  outDict = {'prev': None, 'df': None, 'ts': None}
  if len(dfCond) == 0:
    return None

  ind0 = list(dfCond.index)[0]
  if ind0 == 0:
    return None

  indR = range(ind0 - 1, ind0 + 1)
  dfClose = df.iloc[indR]
  times = dfClose[timeCol].values
  conds = dfClose[usageCol].values
  delta = times[1] - times[0]
  secs = delta / np.timedelta64(1, 's')

  if secs < timeGap:
    prev = conds[0]
  else:
    prev = 'BLANK'
  outDict['prev'] = prev
  outDict['df'] = dfClose
  outDict['ts'] = times[1]

  return outDict

'''
df = GenUsageDf_forTesting()
df = df[df['user_id'] == '1']
df = df.sort_values(['time'])
Mark(df)

condCols = ['prod', 'form_factor']
condValues = [browsingFeat, None]

FindPrev0(
    df,
    condCols=condCols,
    condValues=condValues,
    usageCol='prod',
    userCol='user_id',
    timeCol='time',
    timeGap=10*600)
'''


## find previous activity for a user given conditions:
# condValues for columns: condCols e.g. (product, form_factor)
# it can also find the previous event for a second occurrence of the conditions too
# this happens if secondUsage == True
# then it finds the entry point to the second occurrence
# it requires at least a secondUsageGap for considering the event
# a second usage
def FindPrev(df,
             user,
             condCols,
             condValues,
             userCol='user_id',
             timeCol='ts',
             timeGap=10 * 60,
             secondUsage=False,
             secondUsageGap=3600):

  outDict = {
      'prev': None,
      'df': None,
      'ts': None,
      'prev2': None,
      'df2': None,
      'ts2': None}

  dfUser = df[df[userCol] == user]
  dfUser = Concat_stringColsDf(
      df=dfUser, cols=condCols, colName="cond", sepStr='-')

  out = FindPrev0(
      df=dfUser,
      condCols=condCols,
      condValues=condValues,
      usageCol='cond',
      timeCol=timeCol,
      timeGap=timeGap)

  if out == None:
    return outDict
  else:
    outDict['prev'] = out['prev']
    outDict['df'] = out['df']
    outDict['ts'] = out['ts']

  if secondUsage:
    cond = '-'.join(condValues)
    t1 = outDict['ts']
    t2 = t1 + np.timedelta64(secondUsageGap, 's')

    dfUser2 = dfUser[dfUser[timeCol] > t2]

    out2 = None
    if len(dfUser2) > 0:
      conds = dfUser2['cond'].values
      first = next((i for i, v in enumerate(conds) if v != cond), -1)

      if first < len(dfUser2):
        dfUser2 = dfUser2.iloc[first:]
      else:
        return outDict
      out2 = FindPrev0(
          df=dfUser2,
          condCols=condCols,
          condValues=condValues,
          usageCol='cond',
          userCol=userCol,
          timeCol=timeCol,
          timeGap=timeGap)
    if out2 == None:
      return outDict
    else:
      outDict['prev2'] = out2['prev']
      outDict['ts2'] = out2['ts']
      outDict['df2'] = out2['df']

  return outDict

'''
df = GenUsageDf_forTesting()

FindPrev(df=df,
         user='1',
         condCols=['prod', 'form_factor'],
         condValues=['locFeat', 'COMP'],
         userCol='user_id',
         timeCol='time',
         timeGap=10 * 60,
         secondUsage=True,
         secondUsageGap=1)
'''

## this finds the previous activity of  users satisfying a condition
# e.g. (product, form_factor)
def FindPrevUsers(
    users,
    dfDetails,
    condCols,
    condValues,
    userCol,
    timeCol,
    timeGap,
    secondUsage,
    secondUsageGap=3600):

  out = {'prev': [], 'prev2': []}

  def F(user):
    res = FindPrev(
        df=dfDetails,
        user=user,
        condCols=condCols,
        condValues=condValues,
        userCol=userCol,
        timeCol=timeCol,
        timeGap=timeGap,
        secondUsage=secondUsage,
        secondUsageGap=secondUsageGap)

    prev = res['prev']
    prev2 = res['prev2']
    return {'prev': prev, 'prev2': prev2}

  if len(users) == 0:
    return out
  res = [F(u) for u in users]

  df = pd.DataFrame(res)
  outPrev = df['prev'].values
  outPrev2 = df['prev2'].values
  return {'prev': outPrev, 'prev2': outPrev2}

'''
df = GenUsageDf_forTesting()

Mark(df[df['user_id'].isin(['0', '1', '2'])].sort_values(['user_id', 'time']))

FindPrevUsers(
    users=map(lambda x: str(x), range(10)),
    dfDetails=df,
    condCols=['prod', 'form_factor'],
    condValues=['PresFeat', 'PHN'],
    userCol='user_id',
    timeCol='time',
    timeGap=6000,
    secondUsage=True,
    secondUsageGap=3)

'''


## this first segments the users who satisfy conditions (condValues)
# using two datetime variables to ex and new users
# then it finds out the previous activity for each user
# it returns the previous activity in the same format as the conditions
def FindPrevControlTreat(
    dfSummary,
    dfDetails,
    dt1,
    dt2,
    userCol,
    condCols,
    condValues,
    timeColSumm,
    timeColDet,
    timeGap,
    secondUsage,
    secondUsageGap=3600,
    limit_exUsersNum=200):

  # this subsets a data frame using conditions and then segments it using datetimes
  def SegmentNewUsage(df, dt1, dt2, condCols, condValues, timeCol):
    condDict = {}
    for i in range(len(condCols)):
      col = condCols[i]
      condDict[col] = [condValues[i]]
    ind = BuildCondInd(df, condDict=condDict)
    dfNew = df[(df[timeCol] >= dt1) * ind]
    dfEx = df[(df[timeCol] <=  dt2) * ind]
    outDict = {'new': dfNew, 'ex': dfEx}
    return(outDict)

  ## segment the summary data (dfSummary) using dt1, dt2
  dfDict = SegmentNewUsage(
      df=dfSummary,
      dt1=dt1,
      dt2=dt2,
      condCols=condCols,
      condValues=condValues,
      timeCol=timeColSumm)

  dfNew = dfDict['new']
  dfEx = dfDict['ex']
  newUsers = list(set(dfNew[userCol].values))
  exUsers = list(set(dfEx[userCol].values))
  ## limiting the number of ex-users
  if (limit_exUsersNum is not None) and (len(exUsers) > 200):
    exUsers = exUsers[:limit_exUsersNum]

  new = FindPrevUsers(
      users=newUsers,
      dfDetails=dfDetails,
      condCols=condCols,
      condValues=condValues,
      userCol=userCol,
      timeCol=timeColDet,
      timeGap=timeGap,
      secondUsage=secondUsage,
      secondUsageGap=secondUsageGap)

  ex = FindPrevUsers(
      users=exUsers,
      dfDetails=dfDetails,
      condCols=condCols,
      condValues=condValues,
      userCol=userCol,
      timeCol=timeColDet,
      timeGap=timeGap,
      secondUsage=secondUsage,
      secondUsageGap=secondUsageGap)

  outDict = {'new': None, 'ex': None, 'new2': None, 'ex2': None, 'ss': None}
  prevProdNew = new['prev']
  prevProdEx = ex['prev']
  prevProdNew = [j for i, j in enumerate(prevProdNew) if (j is not None)]
  prevProdEx = [j for i, j in enumerate(prevProdEx) if (j is not None)]
  tabNew = pd.Series(prevProdNew).value_counts()
  tabEx = pd.Series(prevProdEx).value_counts()
  ss = {'new': len(prevProdNew), 'ex': len(prevProdEx)}
  outDict['new'] = tabNew
  outDict['ex'] = tabEx

  if secondUsage == True:
    prevProdNew2 = new['prev2']
    prevProdEx2 = ex['prev2']
    prevProdNew2 = [j for i, j in enumerate(prevProdNew2) if (j is not None)]
    prevProdEx2 = [j for i, j in enumerate(prevProdEx2) if (j is not None)]
    tabNew2 = pd.Series(prevProdNew2).value_counts()
    tabEx2 = pd.Series(prevProdEx2).value_counts()
    outDict['new2'] = tabNew2
    outDict['ex2'] = tabEx2
    ss['new2'] = len(prevProdNew2)
    ss['ex2'] = len(prevProdEx2)
  outDict['ss'] = ss

  return outDict


## compare the entry points for various users
def CompareEntryPoints(
    dfSummary,
    dfDetails,
    dt1,
    dt2,
    userCol,
    condCols,
    condValues,
    timeColSumm,
    timeColDet,
    timeGap,
    treat,
    base,
    otherArms=[],
    colorListOther=[],
    secondUsageGap=3600,
    includePvalue=True,
    removeCols=None,
    removeValues=None):

  if removeCols != None:
    for i in range(len(removeCols)):
      col = removeCols[i]
      values = removeValues[i]
      dfDetails = dfDetails[~dfDetails[col].isin(values)]

  dfDetails = dfDetails.reset_index(drop=True)
  dfSummary = dfSummary.reset_index(drop=True)

  secondUsage = False
  allArms = [base, treat] + otherArms

  if ('new2' in allArms) or ('ex2' in allArms):
    secondUsage = True

  res = FindPrevControlTreat(
      dfSummary=dfSummary,
      dfDetails=dfDetails,
      dt1=dt1,
      dt2=dt2,
      userCol=userCol,
      condCols=condCols,
      condValues=condValues,
      timeColSumm=timeColSumm,
      timeColDet=timeColDet,
      timeGap=timeGap,
      secondUsage=secondUsage)
  tabTreat = res[treat]
  tabBase = res[base]
  ss = res['ss']
  tabDict = {treat: tabTreat, base: tabBase}
  for i in range(len(otherArms)):
    tabDict[otherArms[i]] = res[otherArms[i]]
  tab = MergeTablesDict(tabDict)
  condName = '-'.join(condValues)
  p = None

  if (ss[base]*ss[treat]) > 0:
    tab2 = TabComplPvalue(
        tab=tab,
        categCol='categ',
        freqCols=['freq_' + treat, 'freq_' + base])
    tab['(1-pvalue)%'] = 100.0 - 100.0 * tab2['p-value']
    ind = ((tab['prop_' + treat] >2) + (tab['prop_' + base] > 2)) > 0
    tab = tab[ind]

    def ExtraFcn1(ax):
      plt.axhline(y=95, alpha=0.3, color='gray', linestyle='--')
      plt.axhline(y=90, alpha=0.3, color='gray', linestyle='--')
      ax.text(0, 95, '95%', fontsize=10)
      ax.text(0, 90, '90%', fontsize=10)

    def ExtraFcn2(ax):
      plt.axvline(x=95, alpha=0.3, color='gray', linestyle='--')
      plt.axvline(x=90, alpha=0.3, color='gray', linestyle='--')
      ax.text(95, 0, '95%', fontsize=10)
      ax.text(90, 0, '90%', fontsize=10)
    cols = ['prop_' + treat, 'prop_' + base]
    for i in range(len(otherArms)):
      cols = cols + [('prop_' + otherArms[i])]
    colorList = ['g', 'r']
    alphaList = [0.6] * 2
    if (len(otherArms) > 0):
      alphaList = alphaList + [0.3] * len(otherArms)
      if (len(colorListOther) == 0):
        colorListOther = ['gray'] * len(otherArms)
      colorList = colorList + colorListOther
    if (includePvalue):
      cols = cols + ['(1-pvalue)%']
      colorList = colorList + ['y']
      alphaList = alphaList + [0.2]

    p = PltCols_wrtIndex(
        df=tab,
        cols=cols,
        categCol='categ',
        pltTitle=('' + condName + '; ss: ' + treat + ':' + str(ss[treat]) + ', '
                  + base + ':' + str(ss[base])),
        colorList=colorList,
        alphaList=alphaList,
        ymax=100,
        ExtraFcn=ExtraFcn2,
        orient='h')

  return {'tab': tab, 'plot': p}


## plots the bar plot of the time difference between two time columns
def Plt_timeDiff_barPlot(df, col1, col2, pltTitle=None):

  x = df[col2] - df[col1]
  y = x.dt.days
  if pltTitle == None:
    pltTitle = col2 + ' - ' + col1
  CutBarPlot(y, method='uniform', num=5, pltTitle=pltTitle)
