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

"""This is a library to calculate metrics for user analysis
This is dependent on data_analysis.py, and that needs to be sourced as well
The functions in this library are focused on calculating "per item metrics"
for example for these metrics:
duration per user per date:  item = [user, date]
count per user: item = user

Here is an explanation of the input to most of the functions
itemCols: these are the columns which identify an item.
e.g. itemCols = ['user_id', 'date']  or itemCols = ['user_id']
usageCols: these are the usages we are interested in
e.g. usageCols = ["feature"] means
we would like to compare/calculate usage of features
e.g. usageCols = ["surface", product"] means
we would like to compare product usages across surfaces
e.g. we the usages could look like:
someSurface-mailingFeat, someSurface-search, randSurface-socialFeat,
otherSurface-mailingFeat
sliceCols are the slices of data we are interested in
for example sliceCols = [country, gender] or sliceCols = [expt]


The code has these parts
PART 1: calculates per item metrics but does not provide confidence intervals
(except for penetration)
PART 2: calculates per item metrics with distributions
in the slice and confidence intervals for the means
as well as confidence intervals for difference of means
PART 3: functions to partition data
e.g. only take subset of data where the users have watchFeat at least once
during the whole period
"""

######### PART 1: per item metrics w/o CI except for penetration
## calculate number of distinct items (e.g. users) in a slice
# for example slice:[country, date]
def CountItems_perSlice(df, itemCols, sliceCols, newColName='item_num'):

  df2 = df.copy()

  # adding a single item col which captures all itemCols
  df2 = Concat_stringColsDf(
      df=df2,
      cols=itemCols,
      colName='item_comb',
      sepStr='-')

  df2 = df2[['item_comb'] + sliceCols]
  g = df2.groupby(sliceCols, as_index=False)
  df3 = g.agg(lambda x: len(set(x)))

  if newColName is not None:
    df3.columns = sliceCols + [newColName]
  df3 = df3.reset_index(drop=True)

  return df3

'''
df = GenUsageDf_forTesting()
CountItems_perSlice(
    df=df,
    itemCols=['user_id', 'prod'],
    sliceCols=['country', 'date'],
    newColName='slice_num_items')
'''

## finds the total number of items (e.g. users) per slice
# finds the total (e.g. duration or money) given in valueCol
# for a usage (usageCols) and num of usages per item (user)
# then merges the two data sets to calculate per item (e.g. user/day) metrics
# IMPORTANT: total number of items is calculated per slice,
# not per slice and usage
# so we are not calculating the usage count mean for active users only
# its calculated for active and non-active users
# another way to do this is:
# add an occ column to data df[ , 'usage_occ'] = 1
# for each slice, if usage occurs for some item
# but not for others for that slice
# add slice, item, usage to data and assign the usage_occ to be zero
# adjustment is possible for denominators
def Calc_perItemMetrics(
    df,
    itemCols,
    sliceCols,
    usageCols,
    valueCol=None,
    occColName="occ",
    adjustDenom=None):

  valueColInit = valueCol
  if valueCol is None:
    valueCol = "dummy"
    df[valueCol] = 1
  ## adding an occurrence column
  df[occColName] = 1

  # adding a single column to capture all itemCols dimensions
  df2 = Concat_stringColsDf(
      df=df.copy(),
      cols=itemCols,
      colName='item',
      sepStr='-')

  # calculate total numbers of items (e.g. users) per slice
  dfItemCount = CountItems_perSlice(
      df2,
      itemCols=['item'],
      sliceCols=sliceCols,
      newColName='item_count_in_slice')

  # here we adjust denoms if needed
  if adjustDenom == "max":
    Mark("denom adjustment was invoked.")
    dfItemCount["item_count_in_slice"] = dfItemCount[
        "item_count_in_slice"].max()

  ## calculating total count and value per item.
  g = df2.groupby(sliceCols + usageCols, as_index=False)
  aggFcnDict = {
      'item':{'_with_usage_count': lambda x: len(set(x))},
      occColName:{'_count': sum},
      valueCol:{'_total': sum}}

  dfAgg = g.agg(aggFcnDict)
  dfAgg.columns = [''.join(col).strip() for col in dfAgg.columns.values]

  dfM = pd.merge(dfAgg, dfItemCount, how='left')

  metrics = [occColName + '_count', valueCol + '_total']

  for metric in metrics:
    dfM[metric + '_per_item'] = (
        1.0 * dfM[metric] /
        dfM['item_count_in_slice']).map(Signif(3))

  dfM['penetration'] = (
      1.0 * dfM['item_with_usage_count'] /
      dfM['item_count_in_slice']).map(Signif(5))

  if valueColInit is None:
    for col in dfM.columns:
      if col.startswith('dummy'):
        del dfM[col]

  return dfM

'''
df = GenUsageDf_forTesting()

## count user/date
Calc_perItemMetrics(
    df=df,
    itemCols=['user_id', 'date'],
    sliceCols=['country'],
    usageCols=['prod'],
    valueCol='value')
'''

## calculate the sd for
# bernouli distbn avg even when the sample size is small
def CalcBernouliSd(p, ss):

  if ss == 0:
    return float('nan')

  sd = math.sqrt(1.0 * p * (1 - p) / ss)

  if ss < 20:
     sd = 0.5*(sd + math.sqrt(0.5 * 0.5 / ss))

  if ss < 5:
    sd = math.sqrt(0.5 * 0.5 / ss)

  return sd

'''
Mark(CalcBernouliSd(p=0.8, ss=5))
Mark(CalcBernouliSd(p=0.8, ss=20))
Mark(CalcBernouliSd(p=0.8, ss=30))
Mark(CalcBernouliSd(p=0.99, ss=4))
Mark(CalcBernouliSd(p=0.99, ss=5))
'''

## calculates the penetration with conf intervals
# this is done via the fast method
# (which does not complete the data)
# since for Bernouli its possible to calc sd
def CalcItemPenet(
    df,
    itemCols,
    sliceCols,
    usageCols,
    adjustDenom=None,
    pltIt=False):

  # drop repeated rows to only count a usage once
  cols = itemCols + sliceCols + usageCols
  Mark(df.shape, "df.shape before removing duplicates")
  df = df[cols].drop_duplicates()
  Mark(df.shape, "df.shape after removing duplicates")

  perItemDf = Calc_perItemMetrics(
      df=df,
      itemCols=itemCols,
      sliceCols=sliceCols,
      usageCols=usageCols,
      valueCol=None,
      adjustDenom=adjustDenom)

  perItemDf2 = perItemDf[
      sliceCols + usageCols + ['item_count_in_slice', 'penetration']].copy()
  perItemDf2.columns = sliceCols + usageCols + ['ss', 'penetration']
  perItemDf2['penetration_sd'] = perItemDf2.apply(
      lambda row: CalcBernouliSd(p=row['penetration'], ss=row['ss']) , axis=1)

  perItemDf2['penetration_ci_lower'] = (
      perItemDf2['penetration'] -
      2 * perItemDf2['penetration_sd']).map(lambda x: max(x, 0.0))
  perItemDf2['penetration_ci_upper'] = (
      perItemDf2['penetration'] +
      2 * perItemDf2['penetration_sd']).map(lambda x: min(x, 1.0))
  perItemDf3 = perItemDf2[
      sliceCols +
      usageCols +
      ['penetration', 'penetration_sd', 'penetration_ci_lower',
      'penetration_ci_upper']].copy()


  for col in ['penetration', 'penetration_sd', 'penetration_ci_lower',
             'penetration_ci_upper']:
    perItemDf3[col] = (100 * perItemDf3[col]).map(Signif(3))

  if pltIt:
    perItemDf3 = Concat_stringColsDf(
        df=perItemDf3,
        cols=usageCols,
        colName='usage_combin',
        sepStr='-')

    PlotCIWrt(
        df=perItemDf3,
        colUpper='penetration_ci_upper',
        colLower='penetration_ci_lower',
        sliceCols=sliceCols,
        labelCol='usage_combin',
        col=None,
        ciHeight=0.5,
        rotation=0,
        addVerLines=[],
        logScale=False,
        lowerLim=None,
        pltTitle='',
        figSize=[5, 20])

  return perItemDf3

'''
df = GenUsageDf_forTesting()
CalcItemPenet(df=df,
              itemCols=['user_id', 'date'],
              sliceCols=['country'],
              usageCols=['prod'])
'''

## we compare, total usage for items, number of usages and
# percent of items in slice who has usage (e.g. app usage) aka penetration
# this calls: Calc_perItemMetrics to calculate metrics
def CompareUsageSlices(
    df,
    usageCol,
    compareCol,
    itemCols,
    valueCol,
    compareValues=None,
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    SubDfPost=None,
    extraCols=[],
    pltTitlePre='',
    sizeAlpha=0.75):

  if itemColsPltTitle is None:
    itemColsPltTitle = '_'.join(itemCols)

  df2 = df.copy()
  usageCols = [usageCol] + extraCols

  if condDictPre is not None:
    ind = BuildCondInd(df=df2, condDict=condDictPre)
    df2 = df2[ind].copy()

  df2 = df2.reset_index(drop=True)
  dfMetrics = Calc_perItemMetrics(
      df=df2,
      itemCols=itemCols,
      sliceCols=[compareCol],
      usageCols=usageCols,
      valueCol=valueCol)

  if condDictPost is not None:
    ind = BuildCondInd(df=dfMetrics, condDict=condDictPost)
    dfMetrics = dfMetrics[ind].copy()

  if SubDfPost is not None:
    dfMetrics = SubDfPost(dfMetrics)
  valueCol2 = 'occ_count_per_item'

  #Mark(dfMetrics[:3])
  outDf = PivotPlotWrt(
      df=dfMetrics,
      pivotIndCol=usageCol,
      compareCol=compareCol,
      valueCol=valueCol2,
      cols=compareValues,
      pltTitle=pltTitlePre + 'usage_count_per_' + itemColsPltTitle,
      sizeAlpha=sizeAlpha)['df']
  Mark(outDf[:2])

  valueCol2 = valueCol + '_total_per_item'
  outDf = PivotPlotWrt(
      df=dfMetrics,
      pivotIndCol=usageCol,
      compareCol=compareCol,
      valueCol=valueCol2,
      cols=compareValues,
      pltTitle=pltTitlePre + valueCol + '_total_per_' + itemColsPltTitle,
      sizeAlpha=sizeAlpha)['df']

  valueCol = 'penetration'

  outDf = PivotPlotWrt(
      df=dfMetrics,
      pivotIndCol=usageCol,
      compareCol=compareCol,
      valueCol=valueCol,
      cols=compareValues,
      pltTitle=pltTitlePre + 'percent_penetration_' + itemColsPltTitle,
      sizeAlpha=sizeAlpha)['df']

  return dfMetrics

'''
df = GenUsageDf_forTesting()

itemCols = ['user_id', 'date']
compareCol = 'country'
usageCol = 'prod'
valueCol = 'duration'

dfMetrics = CompareUsageSlices(
    df,
    usageCol,
    compareCol,
    itemCols,
    valueCol,
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    SubDfPost=None,
    extraCols=[],
    pltTitlePre='',
    sizeAlpha=0.75)


## we boost search and docs on expt arm
df = GenUsageData_withExpt(
    userIdsPair=[range(100), range(101, 200)],
    dt1=datetime.datetime(2017, 4, 12),
    dt2=datetime.datetime(2017, 4, 14),
    timeGridLenPair=['2h', '2h'],
    durLimitPair = [3600, 3000],
    prodsChoicePair = [
        [browsingFeat, 'mailingFeat',
         'editingFeat', 'exploreFeat', 'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat'],
        [browsingFeat, 'mailingFeat',
         'editingFeat', 'exploreFeat', 'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat', browsingFeat, 'editingFeat']])

itemCols = ['user_id', 'date']
compareCol = 'expt'
usageCol = 'prod'
valueCol = 'dur_secs'

dfMetrics = CompareUsageSlices(
    df=df,
    usageCol=usageCol,
    compareCol=compareCol,
    itemCols=itemCols,
    valueCol=valueCol,
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    SubDfPost=None,
    extraCols=[],
    pltTitlePre='',
    sizeAlpha=0.75)
'''


############ PART 2: metrics with distributions for each slice
# and confidence intervals for means and mean differences


## fill in usage data, when there is no usage for an item,
# an assigning valueCols to be according to missingReplaceDict
# this is done so we could calculate per item metrics summary values
# e.g. mean, quantiles, SD
# this could be avoided by complex book keeping, but not practical often
# a missing usage on an item typically means ZERO usage
# for that [item, usage] combination
# occurrence column is added to keep track of actual observed usages,
# occ=0 means no usage occurrence
# if allItemCombins is True the returned df has
# all the possible item combinations
# e.g. if itemCols = [user_id, date], it will add
# all dates for a given user
def CompleteDataGrid(
    df,
    itemCols,
    usageCols,
    valueCols,
    missingReplaceDict,
    allItemCombins=False,
    occColName="occ"):

  # we create a new column to keep track of real occurrences of usage
  df[occColName] = 1

  gridCols = itemCols + usageCols
  combined = []
  for col in gridCols:
    l = df[col].values
    l = UniqueList(l)
    combined.append(l)

  gridDf = pd.DataFrame(
      columns=gridCols,
      data=list(itertools.product(*combined)))

  dfM = pd.merge(
      gridDf, df[itemCols + usageCols + [occColName] + valueCols], how='left')

  # assigning occurrence to zero for usages we added
  # (because they are not real usage occurrences).
  dfM.loc[dfM[occColName].isnull(), occColName] = 0

  for col in valueCols:
    if missingReplaceDict is not None and len(missingReplaceDict) > 0:
      fillValue = missingReplaceDict[col]
    dfM.loc[dfM[col].isnull(), col] = fillValue

  if allItemCombins:
    return dfM

  ## when we only want the item combinations appearing in data
  df0 = df[itemCols]
  # we only need one copy of each item
  df0 = df0.drop_duplicates()
  df0 = df0.reset_index(drop=True)
  # we only keep item combinations which occurred in real data
  dfM2 = pd.merge(df0, dfM, on=itemCols, how='left')

  return dfM2

'''
df = GenUsageDf_forTesting()
itemCols = ['user_id', 'date']
sliceCols = ['country']
usageCols = ['prod']
valueCols = ['value']
missingReplaceDict = {'value':0}
filledDf1 = CompleteDataGrid(
    df=df,
    itemCols=itemCols,
    usageCols=usageCols,
    valueCols=valueCols,
    missingReplaceDict=missingReplaceDict,
    allItemCombins=False)

Mark(filledDf1.shape, text='only existing item combinations allowed')
Mark(filledDf1)

filledDf2 = CompleteDataGrid(
    df=df,
    itemCols=itemCols,
    usageCols=usageCols,
    valueCols=valueCols,
    missingReplaceDict=missingReplaceDict,
    allItemCombins=True)

Mark(filledDf2.shape, text='all item combinations are added')
Mark(filledDf2)
'''

## fills in no usage data, for each grid separately
# if fillingScope='bySlice' or globally if fillingScope='global'
def CompleteDataGrid_withSlices(
    df,
    itemCols,
    usageCols,
    valueCols,
    sliceCols,
    missingReplaceDict,
    fillingScope='global',
    allItemCombins=False,
    itemSliceMatchCols=None,
    occColName="occ"):

  if sliceCols is None or len(sliceCols) == 0:
    dfM = CompleteDataGrid(
      df=df,
      itemCols=itemCols,
      usageCols=usageCols,
      valueCols=valueCols,
      missingReplaceDict=missingReplaceDict,
      allItemCombins=allItemCombins,
      occColName=occColName)
    return dfM

  ## this will add all usage combinations occurred in data
  # item combinations is decided by allItemCombins
  if fillingScope == 'global':
    dfM = CompleteDataGrid(
      df=df,
      itemCols=itemCols + sliceCols,
      usageCols=usageCols,
      valueCols=valueCols,
      missingReplaceDict=missingReplaceDict,
      allItemCombins=allItemCombins,
      occColName=occColName)

    if itemSliceMatchCols is None:
      return dfM

    ## we need to restrict the slice possibilities for each item
    # we only require a match on itemSliceMatchCols
    df0 = df[itemSliceMatchCols + sliceCols]
    # we only need one copy of each item
    df0 = df0.drop_duplicates()
    df0 = df0.reset_index(drop=True)
    # we only keep item/slice combinations
    # which have a match between itemSliceMatchCols and sliceCols
    # for example for: itemCols = [user_id, date],
    # sliceCols = [country], itemSliceMatchCols=[user_id]
    # we keep (user, date, country, usage) if the user was seen in country
    dfM2 = pd.merge(dfM, df0, on=itemSliceMatchCols + sliceCols, how='inner')

    return dfM2

  if fillingScope == 'bySlice':
    df1 = df[sliceCols + itemCols + usageCols + valueCols].copy()

    def F(group):
      dfM = CompleteDataGrid(
          df=group,
          itemCols=itemCols,
          usageCols=usageCols,
          valueCols=valueCols,
          missingReplaceDict=missingReplaceDict,
          allItemCombins=allItemCombins,
          occColName=occColName)

      for col in sliceCols:
        dfM[col] = group[col].values[0]

      dfM = dfM.reset_index(drop=True)
      return(dfM)

    g = df1.groupby(sliceCols, as_index=False)

    outDf = g.apply(F)
    outDf = outDf.reset_index(drop=True)
    return outDf

  print("No implemented fillingScope was found. Returning None.")
  return None

'''
df = GenUsageDf_forTesting()

itemCols = ['user_id', 'date']
sliceCols = ['country']
usageCols = ['prod']
valueCols = ['value']
missingReplaceDict = {'value':0}
filledDf1 = CompleteDataGrid_withSlices(
  df=df,
  itemCols=itemCols,
  usageCols=usageCols,
  valueCols=valueCols,
  sliceCols=sliceCols,
  missingReplaceDict={'value':0},
  fillingScope='bySlice',
  allItemCombins=False)
Mark(filledDf1.shape, text='filling done by slice,  existing item combin only')
Mark(filledDf1[:20])

filledDf2 = CompleteDataGrid_withSlices(
  df=df,
  itemCols=itemCols,
  usageCols=usageCols,
  valueCols=valueCols,
  sliceCols=sliceCols,
  missingReplaceDict={'value':0},
  fillingScope='bySlice',
  allItemCombins=True)
Mark(filledDf2.shape, text='filling done by slice, all item combin')
Mark(filledDf2[:20])

filledDf3 = CompleteDataGrid_withSlices(
  df=df,
  itemCols=itemCols,
  usageCols=usageCols,
  valueCols=valueCols,
  sliceCols=sliceCols,
  missingReplaceDict={'value':0},
  fillingScope='global',
  allItemCombins=False)
Mark(filledDf3.shape, text='filling done globally, existing item combin only')
Mark(filledDf3[:20])

filledDf4 = CompleteDataGrid_withSlices(
  df=df,
  itemCols=itemCols,
  usageCols=usageCols,
  valueCols=valueCols,
  sliceCols=sliceCols,
  missingReplaceDict={'value':0},
  fillingScope='global',
  allItemCombins=True)
Mark(filledDf4.shape, text='filling done globally, all item/slice combin')
Mark(filledDf4[:20])

filledDf5 = CompleteDataGrid_withSlices(
  df=df,
  itemCols=itemCols,
  usageCols=usageCols,
  valueCols=valueCols,
  sliceCols=sliceCols,
  missingReplaceDict={'value':0},
  fillingScope='global',
  allItemCombins=True,
  itemSliceMatchCols=["user_id"])
Mark(
    filledDf5.shape,
    text='filling done globally, all item combin, only match on user_id with a slice in data is needed to keep row')
Mark(filledDf5[:20])

filledDf6 = CompleteDataGrid_withSlices(
  df=df,
  itemCols=itemCols,
  usageCols=usageCols,
  valueCols=valueCols,
  sliceCols=sliceCols,
  missingReplaceDict={'value':0},
  fillingScope='global',
  allItemCombins=True,
  itemSliceMatchCols=itemCols)
Mark(filledDf6.shape, text=('filling done globally, all item combin,' +
                            ' (but not item/slice combinations)' +
                            ' we make sure that user_id was seen' +
                            ' in the same slice.' + 'Otherwise we discard'))
Mark(filledDf6[:20])
'''

## orig function
## it provides per item metrics, but also provides quantiles and median
# to have an understanding of the variability
# its slower because it completes the data grid
# itemCols: these are the units for which we aggr data
# in particular valueCol is summed
def Calc_perItemMetrics_withDistr(
    df,
    itemCols,
    sliceCols,
    usageCols,
    valueCol=None,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=False,
    itemSliceMatchCols=None):

  valueCols = [valueCol]

  ## we complete the data grid,
  # by adding zero values (valueCol) for usages which do not appear
  df2 = CompleteDataGrid_withSlices(
    df=df,
    itemCols=itemCols,
    usageCols=usageCols,
    valueCols=valueCols,
    sliceCols=sliceCols,
    missingReplaceDict={valueCol:0},
    fillingScope=fillingScope,
    allItemCombins=allItemCombins,
    itemSliceMatchCols=itemSliceMatchCols,
    occColName=occColName)

  Mark(df.shape, text="This is the size of the original data: ", color="green")
  Mark(df2.shape, text="This is the size of the completed data: ", color="red")

  df2 = Concat_stringColsDf(
      df=df2,
      cols=itemCols,
      colName='item',
      sepStr='-')

  ## aggregate to item level (e.g. session (user/date) level)
  gbCols = sliceCols + usageCols + ['item']

  g = df2.groupby(gbCols, as_index=False)
  aggFcnDict = {
      occColName:{'_count':sum},
      valueCol:{'_total': sum}}
  dfAgg = g.agg(aggFcnDict)
  dfAgg.columns = [''.join(col).strip() for col in dfAgg.columns.values]
  dfAgg.rename(columns={(occColName + '_count'):'usage_count'}, inplace=True)

  ## aggregate to slice, usage level (aggregate item out)
  g2 =  dfAgg.groupby(sliceCols + usageCols, as_index=False)
  aggFcnDict2 = {
      'item':{'_count_in_slice': lambda x: len(set(x))},
      ('usage_count'):{
          '_item': lambda x: sum(x > 0), # num of items with at least one non-zero usage
          '_item_all': lambda x: len(x), # num of all items including with zero usage
          '_mean': np.mean,
          '_median':np.median,
          '_q_0.05':lambda x: np.percentile(a=x, q=5),
          '_Q1':lambda x: np.percentile(a=x, q=25),
          '_Q3':lambda x: np.percentile(a=x, q=75),
          '_q_0.95':lambda x: np.percentile(a=x, q=95),
          '_sd':np.std},
      (valueCol + '_total'):{
          '_mean': np.mean,
          '_median':np.median,
          '_q_0.05':lambda x: np.percentile(a=x, q=5),
          '_Q1':lambda x: np.percentile(a=x, q=25),
          '_Q3':lambda x: np.percentile(a=x, q=75),
          '_q_0.95':lambda x: np.percentile(a=x, q=95),
          '_sd':np.std}}

  dfAgg2 = g2.agg(aggFcnDict2)
  dfAgg2.columns = [''.join(col).strip() for col in dfAgg2.columns.values]

  dfAgg2.rename(
      columns={
          'usage_count_item':'item_with_usage_count',
          'usage_count_item_all':'item_with_wo_usage_count'},
      inplace=True)

  #plt.scatter(dfAgg2['item_with_wo_usage_count'], dfAgg2['item_count_in_slice'])

  ## item penetration of usage (in each slice)
  dfAgg2['penetration'] = (
      1.0 * dfAgg2['item_with_usage_count'] / dfAgg2['item_with_wo_usage_count'])

  dfAgg2['penetration_sd'] = dfAgg2.apply(
      lambda row: CalcBernouliSd(p=row['penetration'], ss=row['item_with_wo_usage_count']) , axis=1)

  dfAgg2['penetration_ci_upper'] = (
      dfAgg2['penetration'] +
      2.0 * dfAgg2['penetration_sd']).map(lambda x: min(x, 1.0))
  dfAgg2['penetration_ci_lower'] = (
      dfAgg2['penetration'] -
      2.0 * dfAgg2['penetration_sd']).map(lambda x: max(x, 0.0))

  ## adding confidence intervals for means:
  # the "n" in the sample average calculation is
  # the number of items with positive counts/values: 'item_with_usage_count'
  # this is probably too conservative since zero values contribute
  # we could instead use: 'item_with_wo_usage_count' which is less conservative
  # to damping the variation: #TODO Reza Hosseini: investigate
  #denomCol = 'item_with_usage_count'
  denomCol = 'item_with_wo_usage_count'
  plt.scatter(dfAgg2['item_with_wo_usage_count'], dfAgg2['item_with_usage_count'])
  for col0 in ['usage_count', valueCol + '_total']:
    dfAgg2[col0 + '_mean_ci_upper'] = (
        dfAgg2[col0 + '_mean'] +
        (2.0 * dfAgg2[col0 + '_sd'] / dfAgg2[denomCol].map(math.sqrt))
        )
    dfAgg2[col0 + '_mean_ci_lower'] = (
        dfAgg2[col0 + '_mean'] -
        (2.0 * dfAgg2[col0 + '_sd'] / dfAgg2[denomCol].map(math.sqrt))
        )

  dfAgg2 = dfAgg2.reset_index(drop=True)

  return dfAgg2

'''
df = GenUsageDf_forTesting()

itemCols = ['user_id', 'date']
sliceCols = ['country']
usageCols = ['prod']
valueCols = ['value']
missingReplaceDict = {'value':0}


## fast method without variability
itemMetricsDf1 = Calc_perItemMetrics(
    df=df,
    itemCols=itemCols,
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol='value')

## slower method but with variability
itemMetricsDf2 = Calc_perItemMetrics_withDistr(
    df=df,
    itemCols=itemCols,
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol="value",
    fillingScope="bySlice",
    allItemCombins=False,
    itemSliceMatchCols=None)

## slower method but with variability
itemMetricsDf3 = Calc_perItemMetrics_withDistr(
    df=df,
    itemCols=itemCols,
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol="value",
    fillingScope="bySlice",
    allItemCombins=True,
    itemSliceMatchCols=None)

## slower method, with global filling, not recommended usually
itemMetricsDf4 = Calc_perItemMetrics_withDistr(
    df=df,
    itemCols=itemCols,
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol="value",
    fillingScope="global",
    allItemCombins=True,
    itemSliceMatchCols=["user_id"])

Mark(
    itemMetricsDf1, 'fast method, but only means')
Mark(
    itemMetricsDf2,
    'slower method but more metrics, bySlice filling, only existing items')
Mark(
    itemMetricsDf3,
    'slower method but more metrics, bySlice filling,
    all items combin within slice')
Mark(
    itemMetricsDf4,
    'slower method but more metrics, global filling,
    all items combin globally, matching slice with user_id only')
'''

## penetration calculated in per item metrics have the caveat of a
# shifting denominator
# to remedy this we could consider for example:
# [num(user, date, prod) / num users in a period]
# in this case itemCols = [user_id, date]
# while the denominator counts number of "units" which are users in this case
# unitCols should be subset of itemCols in general
# for example itemCols = [user_id, date]  and  unitCols=[user_id]
# we count # [item, usage] for each unit
# so for example for one user,
# we find out the number of dates where user used usage:socialFeat
def CalcItemUsageOcc_perUnit_withDistr(
  df,
  itemCols,
  unitCols,
  sliceCols,
  usageCols,
  valueCol=None,
  occColName="occ",
  fillingScope="bySlice",
  allItemCombins=False,
  itemSliceMatchCols=None):

  if valueCol is None:
    valueCol = "dummy"
    df["dummy"] = 1

  valueCols = [valueCol]

  ## we complete the data grid,
  # by adding zero values (valueCol) for usages which do not appear
  filledDf = CompleteDataGrid_withSlices(
    df=df,
    itemCols=itemCols,
    usageCols=usageCols,
    valueCols=valueCols,
    sliceCols=sliceCols,
    missingReplaceDict={valueCol:0},
    fillingScope=fillingScope,
    allItemCombins=allItemCombins,
    itemSliceMatchCols=itemSliceMatchCols,
    occColName=occColName)

  Mark(df.shape, text="This is the size of the original data: ", color="green")
  Mark(filled.shape, text="This is the size of the completed data: ", color="red")

  df2 = Concat_stringColsDf(
    df=filledDf.copy(),
    cols=itemCols,
    colName='item',
    sepStr='-')

  df2 = Concat_stringColsDf(
    df=df2.copy(),
    cols=unitCols,
    colName='unit',
    sepStr='-')

  ## aggregate to item level (e.g. session (user/date) level)
  gbCols = sliceCols + usageCols + ['item', 'unit']
  g = df2.groupby(gbCols, as_index=False)
  aggFcnDict = {
    occColName:{'_count':sum},
    valueCol:{'_total': sum}}
  dfAgg = g.agg(aggFcnDict)
  dfAgg.columns = [''.join(col).strip() for col in dfAgg.columns.values]
  dfAgg.rename(columns={(occColName + '_count'):'usage_count'}, inplace=True)

  ## aggregate to unit, slice, usage level
  # for each unit and usage we get number of items with that usage
  g2 =  dfAgg.groupby(sliceCols + usageCols + ['unit'], as_index=False)
  aggFcnDict2 = {('usage_count'):{'_item': lambda x: sum(x > 0)}}
  dfAgg2 = g2.agg(aggFcnDict2)
  dfAgg2.columns = [''.join(col).strip() for col in dfAgg2.columns.values]
  dfAgg2.rename(columns={'usage_count_item':'item_with_usage_count'}, inplace=True)

  g3 =  dfAgg2.groupby(sliceCols + usageCols, as_index=False)
  aggFcnDict3 = {
    'unit':{'_count_in_slice': lambda x: len(set(x))},
    'item_with_usage_count':{
        '_mean': np.mean,
        '_median':np.median,
        '_Q1':lambda x: np.percentile(a=x, q=25),
        '_Q3':lambda x: np.percentile(a=x, q=75),
        '_sd':np.std}}

  dfAgg3 = g3.agg(aggFcnDict3)
  dfAgg3.columns = [''.join(col).strip() for col in dfAgg3.columns.values]

  ## adding confidence intervals for means:
  col0 = 'item_with_usage_count'
  dfAgg3[col0 + '_mean_ci_upper'] = (
    dfAgg3[col0 + '_mean'] +
    (2.2 * dfAgg3[col0 + '_sd'] / dfAgg3['unit_count_in_slice'].map(math.sqrt))
    )
  dfAgg3[col0 + '_mean_ci_lower'] = (
    dfAgg3[col0 + '_mean'] -
    (2.2 * dfAgg3[col0 + '_sd'] / dfAgg3['unit_count_in_slice'].map(math.sqrt))
    )

  dfAgg3 = dfAgg3.reset_index(drop=True)

  return dfAgg3

'''
df = GenUsageDf_forTesting()
itemCols = ['user_id', 'date']
sliceCols = ['country']
usageCols = ['prod']
valueCols = ['value']
missingReplaceDict = {'value':0}

#Mark(df)
res1 = CalcItemUsageOcc_perUnit_withDistr(
    df=df,
    itemCols=['user_id', 'date'],
    unitCols=['user_id'],
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol=None,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=False,
    itemSliceMatchCols=None)
Mark(res1[:20], "bySlice, only existing item combin")

res2 = CalcItemUsageOcc_perUnit_withDistr(
    df=df,
    itemCols=['user_id', 'date'],
    unitCols=['user_id'],
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol=None,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=True,
    itemSliceMatchCols=None)
Mark(res2[:20], "bySlice, all item combin")

res3 = CalcItemUsageOcc_perUnit_withDistr(
    df=df,
    itemCols=['user_id', 'date'],
    unitCols=['user_id'],
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol=None,
    occColName="occ",
    fillingScope="global",
    allItemCombins=False,
    itemSliceMatchCols=None)
Mark(res3[:20], "global, only existing item combin")

res4 = CalcItemUsageOcc_perUnit_withDistr(
    df=df,
    itemCols=['user_id', 'date'],
    unitCols=['user_id'],
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol=None,
    occColName="occ",
    fillingScope="global",
    allItemCombins=True,
    itemSliceMatchCols=None)
Mark(res4[:20], "global, all item/slice combin: this is what you shouldn't do normally")

res5 = CalcItemUsageOcc_perUnit_withDistr(
    df=df,
    itemCols=['user_id', 'date'],
    unitCols=['user_id'],
    sliceCols=sliceCols,
    usageCols=usageCols,
    valueCol=None,
    occColName="occ",
    fillingScope="global",
    allItemCombins=True,
    itemSliceMatchCols=["user_id"])
Mark(res5[:20], "global, all item combin, all items combin globally, matching slice with user_id only")
'''

## compares usage distributions
# we provide plots
def CompareUsageSlices_withDistr(
    df,
    usageCol,
    compareCol,
    itemCols,
    valueCol,
    unitCols=None,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=True,
    itemSliceMatchCols=None,
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    regDictPre=None,
    regDictPost=None,
    SubDfPost=None,
    extraCols=[],
    logScale=True,
    lowerLim=0.0005,
    pltTitlePre='',
    sizeAlpha=0.75,
    figSize=[8, 20]):

  if itemColsPltTitle is None:
    itemColsPltTitle = '_'.join(itemCols)

  df2 = df
  usageCols = [usageCol] + extraCols

  if condDictPre is not None:
    ind = BuildCondInd(df=df2, condDict=condDictPre)
    df2 = df2[ind].copy()
    df2 = df2.reset_index(drop=True)

  if regDictPre is not None:
    df2 = df2[BuildRegexInd(df=df2, regDict=regDictPre)]
    df2 = df2.reset_index()

  dfMetrics = Calc_perItemMetrics_withDistr(
      df=df2,
      itemCols=itemCols,
      sliceCols=[compareCol],
      usageCols=usageCols,
      valueCol=valueCol,
      occColName=occColName,
      fillingScope=fillingScope,
      allItemCombins=allItemCombins,
      itemSliceMatchCols=itemSliceMatchCols)

  # metrics with average item with usage
  dfMetrics_avgItemNum = None
  if unitCols is not None:
    dfMetrics_avgItemNum = CalcItemUsageOcc_perUnit_withDistr(
        df=df2,
        itemCols=itemCols,
        unitCols=unitCols,
        sliceCols=[compareCol],
        usageCols=usageCols,
        valueCol=None,
        occColName=occColName,
        fillingScope=fillingScope,
        allItemCombins=allItemCombins,
        itemSliceMatchCols=itemSliceMatchCols)

  if condDictPost is not None:
    ind = BuildCondInd(df=dfMetrics, condDict=condDictPost)
    dfMetrics = dfMetrics[ind].copy()
    dfMetrics = dfMetrics.reset_index()
    if unitCols is not None:
      ind = BuildCondInd(df=dfMetrics_avgItemNum, condDict=condDictPost)
      dfMetrics_avgItemNum = dfMetrics_avgItemNum[ind].copy()
      dfMetrics_avgItemNum = dfMetrics_avgItemNum.reset_index()

  if regDictPost is not None:
    dfMetrics = dfMetrics[BuildRegexInd(df=dfMetrics, regDict=regDictPost)]
    dfMetrics = dfMetrics.reset_index()
    if unitCols is not None:
      dfMetrics_avgItemNum = dfMetrics_avgItemNum[
          BuildRegexInd(df=dfMetrics_avgItemNum, regDict=regDictPost)]
      dfMetrics_avgItemNum = dfMetrics_avgItemNum.reset_index()

  if SubDfPost is not None:
    dfMetrics = SubDfPost(dfMetrics)

  valueCol2 = 'usage_count_mean'

  pltDict = {}

  for valueCol2 in ['usage_count', valueCol + '_total']:

    pltDict[valueCol2 + '_distr'] = PlotCIWrt(
        df=dfMetrics.copy(),
        colUpper=valueCol2 + '_q_0.95',
        colLower=valueCol2 + '_q_0.05',
        sliceCols=[compareCol],
        labelCol=usageCol,
        col=valueCol2 + '_mean',
        ciHeight=0.5,
        rotation=0,
        addVerLines=[],
        logScale=logScale,
        lowerLim=lowerLim,
        pltTitle=(pltTitlePre + valueCol2 + '_' + itemColsPltTitle
                  + ' distr (5th percentile to 95th)'),
        figSize=figSize)

  valueCol2 = 'penetration'
  for col in ['penetration', 'penetration_sd', 'penetration_ci_upper',
              'penetration_ci_lower']:
    dfMetrics[col] = 100.0 * dfMetrics[col]

  outDf = PivotPlotWrt(
    df=dfMetrics.copy(),
    pivotIndCol=usageCol,
    compareCol=compareCol,
    valueCol=valueCol2,
    pltTitle=pltTitlePre + 'percent_penetration_' + itemColsPltTitle,
    sizeAlpha=sizeAlpha)['df']

  for valueCol2 in ['usage_count_mean', valueCol + '_total_mean',
                    'penetration']:

    dfMetrics[valueCol2 + '_ci_lower'] = dfMetrics[
        valueCol2 + '_ci_lower'].map(lambda x: max(x, 0.0))

    pltDict[valueCol2 + '_ci'] = PlotCIWrt(
        df=dfMetrics.copy(),
        colUpper=valueCol2 + '_ci_upper',
        colLower=valueCol2 + '_ci_lower',
        sliceCols=[compareCol],
        labelCol=usageCol,
        col=valueCol2 + '',
        ciHeight=0.5,
        rotation=0,
        addVerLines=[],
        logScale=False,
        lowerLim=None,
        pltTitle=(pltTitlePre + valueCol2 + '_' + itemColsPltTitle
                  + ' CI for mean'),
        figSize=figSize)

  if unitCols is not None:
    pltDict['avg_itemWithUsage_num_ci'] = PlotCIWrt(
        df=dfMetrics_avgItemNum.copy(),
        colUpper='item_with_usage_count_mean_ci_upper',
        colLower='item_with_usage_count_mean_ci_lower',
        sliceCols=[compareCol],
        labelCol=usageCol,
        col='item_with_usage_count_mean',
        ciHeight=0.5,
        rotation=0,
        addVerLines=[],
        logScale=False,
        lowerLim=None,
        pltTitle=(pltTitlePre + 'item_with_usage_count' + '_' + itemColsPltTitle
                  + ' CI for mean'),
        figSize=figSize)


  return {'dfMetrics': dfMetrics,
          'dfMetrics_avgItemNum': dfMetrics_avgItemNum,
          'pltDict': pltDict}

'''
df = GenUsageDf_forTesting()
usageCols = ['prod']
valueCol = 'duration'
res = CompareUsageSlices_withDistr(
    df=df,
    usageCol='prod',
    compareCol='expt',
    itemCols=['user_id', 'date'],
    valueCol='duration',
    unitCols=["user_id"],
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=True,
    itemSliceMatchCols=["user_id"],
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    regDictPre=None,
    regDictPost=None,
    SubDfPost=None,
    extraCols=[],
    pltTitlePre='',
    sizeAlpha=0.75,
    figSize=[8, 20])
'''

## gets the means and sds plus sample sizes from
# two distributions
# then calculates a confidence interval for the diff
# simple normal approximation via: s = sqrt(s1^2/n1 + s2^2/n2)
# where (s1, n1) (s2, n2) are sample size for first and second populations
# a full df and a summary df which is more light weight is output
# meanCols and sdCols are the mean columns and sd columns respectively
# ssCols: the sample sizes for each metric
# if ssCols has None we replace each None by a dummy column which has 1 all over
def CalcDiffCi_viaIndivMeanSds(
  df,
  meanCols,
  sdCols,
  compareCol,
  comparePair,
  ssCols,
  sliceCols):

  if None in ssCols:
    df["dummy_ss"] = 1
  ssCols = [(x if x is not None else "dummy_ss") for x in ssCols]
  df2 = df[sliceCols + [compareCol] + meanCols + sdCols + UniqueList(ssCols)].copy()
  ind = BuildCondInd(df=df2, condDict={compareCol: comparePair})
  df2 = df2[ind].copy()

  dfList = []
  for i in range(2):
    dfSub = df2[df2[compareCol] == comparePair[i]]
    dfSub = dfSub[sliceCols + meanCols + sdCols + UniqueList(ssCols)]
    dfSub.columns = (
      sliceCols +
      [x + '_' + comparePair[i] for x in (meanCols + sdCols + UniqueList(ssCols))])
    dfList.append(dfSub.copy())

  diffDf = pd.merge(dfList[0], dfList[1], on=sliceCols, how='outer')

  for i in range(len(meanCols)):
    meanCol = meanCols[i]
    sdCol = sdCols[i]
    ssCol = ssCols[i]
    diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0]] = (
      diffDf[meanCol + '_' + comparePair[1]] -
      diffDf[meanCol + '_' + comparePair[0]])

    diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0] + ' %'] = (
      100.0 * diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0]] /
      diffDf[meanCol + '_' + comparePair[0]])

    diffDf[meanCol + '_diff_sd'] = (
      (diffDf[sdCol + '_' + comparePair[0]]**2 / diffDf[ssCol + '_' + comparePair[0]]) +
      (diffDf[sdCol + '_' + comparePair[1]]**2 / diffDf[ssCol + '_' + comparePair[1]])
    ).map(math.sqrt)

    diffDf[meanCol + '_diff_err'] = 2.0 * diffDf[meanCol + '_diff_sd']
    diffDf[meanCol + '_diff_err %'] = (
      100.0 * diffDf[meanCol + '_diff_err'] /
      diffDf[meanCol + '_' + comparePair[0]])

    diffDf[meanCol + '_diff_upper'] = (
      diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0]] +
      diffDf[meanCol + '_diff_err'])

    diffDf[meanCol + '_diff_lower'] = (
      diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0]] -
      diffDf[meanCol + '_diff_err'])

    diffDf[meanCol + '_diff_is_sig'] = ((
      diffDf[meanCol + '_diff_upper'] *
      diffDf[meanCol + '_diff_lower']) > 0) * diffDf[meanCol + '_diff_upper'].map(np.sign)

    diffDf[meanCol + '_diff_range'] = (
      diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0]].map(Signif(2)).map(str) +
      ' +/- ' +
      diffDf[meanCol + '_diff_err'].map(Signif(2)).map(str))

    diffDf[meanCol + '_diff_range %'] = (
      diffDf[meanCol + '_' + comparePair[1] + '_minus_' + comparePair[0] + ' %'].map(Signif(2)).map(str) +
      ' +/- ' +
      diffDf[meanCol + '_diff_err %'].map(Signif(2)).map(str)) + '(%)'

    ## adding stars
    diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == 1, meanCol + '_diff_range'] = (
      '*+' +
      diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == 1, meanCol + '_diff_range'])

    diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == -1, meanCol + '_diff_range'] = (
      '*' +
      diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == -1, meanCol + '_diff_range'])

    diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == 1, meanCol + '_diff_range %'] = (
      '*+' +
      diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == 1, meanCol + '_diff_range %'])

    diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == -1, meanCol + '_diff_range %'] = (
      '*' +
      diffDf.loc[diffDf[meanCol + '_diff_is_sig'] == -1, meanCol + '_diff_range %'])

    for col in sliceCols:
      diffDf.loc[diffDf[meanCol + '_diff_is_sig'].isin([-1, 1]), col] = (
        "*" + diffDf.loc[diffDf[meanCol + '_diff_is_sig'].isin([-1, 1]), col])

  summDf = diffDf[
    sliceCols +
    [x + '_diff_upper' for x in meanCols] +
    [x + '_diff_lower' for x in meanCols] +
    [x + '_diff_range' for x in meanCols] +
    [x + '_diff_range %' for x in meanCols]]

  briefDf = diffDf[
      sliceCols +
      [x + '_diff_range' for x in meanCols]]

  briefDfPer = diffDf[
    sliceCols +
    [x + '_diff_range %' for x in meanCols]]

  briefDf.columns = (
    sliceCols +
    [x + '....' for x in meanCols])

  briefDfPer.columns = (
    sliceCols +
    [x + '(%) ....' for x in meanCols])

  def ColorCell(val):
    color = 'grey'
    if val.startswith("*+"):
      color = 'green'

    elif val.startswith("*-"):
      color = 'red'
    return 'color: %s' % color

  def ColorCell2(val):
    color = 'white'
    if '*' in val:
        color = 'lightblue'
    return 'background-color: %s' % color

  styledBriefDf = briefDf.style.\
    applymap(ColorCell, subset=[x + '....' for x in meanCols]).\
    applymap(ColorCell2, subset=sliceCols)

  styledBriefDfPer = briefDfPer.style.\
    applymap(ColorCell, subset=[x + '(%) ....' for x in meanCols]).\
    applymap(ColorCell2, subset=sliceCols)

  outDict = {
    'fullDf': diffDf,
    'summDf': summDf,
    'briefDf': briefDf,
    'briefDfPer':briefDfPer,
    'styledBriefDf': styledBriefDf,
    'styledBriefDfPer': styledBriefDfPer}

  return outDict

'''
df = GenUsageDf_forTesting()
usageCols = ['prod']
valueCol = 'duration'

dfMetrics = Calc_perItemMetrics_withDistr(
    df=df,
    itemCols=['user_id', 'date'],
    sliceCols=['country', 'expt'],
    usageCols=usageCols,
    valueCol=valueCol,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=True)
dfM = dfMetrics.reset_index(drop=True)
#Mark(dfM, "dfM")

meanCols = ['usage_count_mean', 'duration_total_mean', 'penetration']
sdCols = ['usage_count_sd', 'duration_total_sd', 'penetration_sd']
ssCols = ['item_with_usage_count', 'item_with_usage_count', None]
compareCol = 'expt'
comparePair = ['base', 'test']
sliceCols = [usageCol]

res = CalcDiffCi_viaIndivMeanSds(
    df=dfM,
    meanCols=meanCols,
    sdCols=sdCols,
    compareCol=compareCol,
    comparePair=comparePair,
    ssCols=ssCols,
    sliceCols=sliceCols)

styledBriefDf = res['styledBriefDf']

print('')
styledBriefDf
'''



'''
## Simulation
# example: we boost search and docs on expt arm
# try your own examples by assigning various prodsChoicePair
# which tells what products should be sampled for each of base, expt arms
ss = 20
df = GenUsageData_withExpt(
    userIdsPair=[range(ss), range(ss + 1, 2 * ss)],
    dt1=datetime.datetime(2017, 4, 12),
    dt2=datetime.datetime(2017, 4, 14),
    timeGridLenPair=['2h', '2h'],
    durLimitPair = [3600, 3000],
    prodsChoicePair = [
        [browsingFeat, 'mailingFeat', 'editingFeat',
         'exploreFeat', 'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat'],
        [browsingFeat, 'mailingFeat', 'editingFeat',
         'exploreFeat', 'photoFeat', 'PresFeat',
         'locFeat', 'StorageFeat', browsingFeat]])


itemCols = ['user_id', 'date']
compareCol = 'expt'
usageCol = 'prod'
valueCol = 'dur_secs'
df['user_id'] = df['user_id'].map(str)
df['date'] = df['date'].map(str)


dfMetrics = CompareUsageSlices(
    df=df,
    usageCol=usageCol,
    compareCol=compareCol,
    itemCols=itemCols,
    valueCol=valueCol,
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    SubDfPost=None,
    extraCols=[],
    pltTitlePre='',
    sizeAlpha=0.25)


res = Calc_perItemMetrics_withDistr(
    df=df,
    itemCols=itemCols,
    sliceCols=[compareCol],
    usageCols=[usageCol],
    valueCol=valueCol,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=False,
    itemSliceMatchCols=None)


res = CompareUsageSlices_withDistr(
    df=df,
    usageCol=usageCol,
    compareCol=compareCol,
    itemCols=itemCols,
    valueCol=valueCol,
    unitCols=None,
    occColName="occ",
    fillingScope="bySlice",
    allItemCombins=True,
    itemSliceMatchCols=None,
    itemColsPltTitle=None,
    condDictPre=None,
    condDictPost=None,
    regDictPre=None,
    regDictPost=None,
    SubDfPost=None,
    extraCols=[],
    pltTitlePre='',
    sizeAlpha=0.75,
    figSize=[4, 8])

dfMetrics = res['dfMetrics']

meanCols = ['usage_count_mean', 'dur_secs_total_mean', 'penetration']
sdCols = ['usage_count_sd', 'dur_secs_total_sd', 'penetration_sd']
ssCols = ['item_with_usage_count', 'item_with_usage_count', None]
compareCol = 'expt'
comparePair = ['base', 'test']
sliceCols = [usageCol]

res = CalcDiffCi_viaIndivMeanSds(
    df=dfMetrics,
    meanCols=meanCols,
    sdCols=sdCols,
    compareCol=compareCol,
    comparePair=comparePair,
    ssCols=ssCols,
    sliceCols=sliceCols)

styledBriefDf = res['styledBriefDf']

print('')
display(styledBriefDf)

'''

############# PART 3: partition data based on usage

# indCols: the columns for which we want the prop to be satisfied at least once
# you can pass the prop function to act on a row or on the df itself
def CheckIndCols_satisfyProp_atleastOnce(
    df, indCols, PropDf=None, PropRow=None, propColName='prop'):

  if PropDf is not None:
    df[propColName] = PropDf(df)
  else:
    df[propColName] = df.apply(PropF, axis=1)

  df2 = df[indCols + [propColName]]
  df2.columns = indCols + [propColName + '_across']
  g = df2.groupby(indCols, as_index=False)
  aggDf = g.agg(sum)
  aggDf[propColName + '_binary'] = aggDf[propColName + '_across'] > 0

  propDf = pd.merge(df, aggDf, on=indCols)
  return {'aggDf':aggDf, 'propDf':propDf}

'''
df = pd.DataFrame(columns=['country', 'user_id', 'date', 'prod'])
df.loc[len(df)] = ['US', '0', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'photoFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'exploreFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'exploreFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'editingFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'editingFeat']

def PropRow(row):
  out = float(row['prod'] == 'editingFeat')
  return(out)
def PropDf(df):
  return(df['prod'] == 'editingFeat')
CheckIndCols_satisfyProp_atleastOnce(
    df=df, indCols=['user_id'], PropRow=PropRow, propColName='prop')
CheckIndCols_satisfyProp_atleastOnce(
    df=df, indCols=['user_id'], PropDf=PropDf, propColName='prop')
'''

## checks for multiple properties
# it expects all to be true
def CheckIndCols_satisfyMultipleProp(
    df, indCols, propDfList, propColNames=None):

  propColNames2 = []
  for i in range(len(propDfList)):
    PropDf = propDfList[i]
    if propColNames is not None:
      propColName = propColNames[i]
    else:
      propColName = "prop" + str(i)

    propColNames2.append(propColName)

    res = CheckIndCols_satisfyProp_atleastOnce(
        df=df, indCols=indCols, PropDf=PropDf, PropRow=None,
        propColName=propColName)

    aggDf1 = res['aggDf']
    if i == 0:
      aggDf = aggDf1
      aggDf['prop_multiple_across'] = aggDf[propColName + '_across'].map(str)
      aggDf['prop_multiple_binary'] = aggDf[propColName + '_binary']
    else:
      aggDf = pd.merge(aggDf, aggDf1, on=indCols)
      aggDf['prop_multiple_binary'] = (
          aggDf['prop_multiple_binary'] * aggDf[propColName + '_binary'])
      aggDf['prop_multiple_across'] = (
          aggDf['prop_multiple_across'] +
          ',' +
          aggDf[propColName + '_across'].map(str))

  propDf = pd.merge(df, aggDf, on=indCols)
  return {'aggDf':aggDf, 'propDf':propDf}

'''
df = pd.DataFrame(columns=['country', 'user_id', 'date', 'prod'])
df.loc[len(df)] = ['US', '0', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'photoFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '0', '2017-04-12', 'exploreFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'PresFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'exploreFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'editingFeat']
df.loc[len(df)] = ['US', '2', '2017-04-12', 'editingFeat']

def PropDf1(df):
  return(df['prod'] == 'editingFeat')

def PropDf2(df):
  return(df['prod'] == 'PresFeat')

CheckIndCols_satisfyMultipleProp(
    df=df,
    indCols=['user_id', 'date'],
    propDfList=[PropDf1, PropDf2],
    propColNames=['has_docs', 'has_slides'])
'''

## generate a data subset given one usage condition
def SubsetDf_hadUsage(df, indCols, usageCol, usage):

  res = CheckIndCols_satisfyProp_atleastOnce(
    df=df,
    indCols=indCols,
    PropDf= PropDfFcn(usageCol=usageCol, usage=usage),
    PropRow=None,
    propColName='had_' + usage + '_act')

  augDf = res['propDf']
  subDf = augDf[augDf['had_' + usage + '_act' + '_binary'] == True]
  aggDf = res['aggDf']

  return subDf

## generate a data subset given multiple usages and requiring all
def SubsetDf_hadUsageList(df, indCols, usageCol, usageList):

  if len(usageList) == 1:
    res = SubsetDf_hadUsage(
        df=df, indCols=indCols, usageCol=usageCol, usage=usageList[0])
    return res

  res = CheckIndCols_satisfyMultipleProp(
      df=df,
      indCols=indCols,
      propDfList=[
          PropDfFcn(usageCol=usageCol, usage=usageList[0]),
          PropDfFcn(usageCol=usageCol, usage=usageList[1])],
      propColNames=['had_' + x + '_act' for x in usageList])

  augDf = res['propDf']
  Mark(augDf[:5])
  subDf = augDf[augDf['prop_multiple_binary'] == True]
  aggDf = res['aggDf']

  return subDf

## partition users to heavy and light
def GetHeavyUsers(df, usageCol, userCol, condDict=None):

  df2 = df.copy()

  if condDict is not None:
    df2 = df2.reset_index()
    ind = BuildCondInd(df=df2, condDict=condDict)
    df2 = df2[ind].copy()

  g = df2[[userCol, usageCol]].groupby([userCol], as_index=False)
  df3 = g.agg(sum)
  df3['userEng'] = CutConti(x=df3[usageCol].values, num=4, method='quantile')
  levels = sorted(set(df3['userEng']))
  print(levels)
  lightLev = levels[0:2]
  df3['heavy'] = ~df3['userEng'].isin(lightLev)
  heavyUsers = set(df3[df3['heavy']][userCol].values)
  lightUsers = set(df3[df3['heavy'] == False][userCol].values)

  return {'heavy': heavyUsers, 'light': lightUsers}

'''
df2 = df.copy()
df2 = df[df['category_standard'].map(str) != 'nan']
df2 = df2[df2['is_imp_categ'] == True]
GetHeavyUsers(
    df=df2,
    usageCol='duration_minutes',
    userCol='gaia_id',
    condDict={'date': ['2017-09-18', '2017-09-19']})
'''

## balancing the sample sizes (in terms of number of items)
# we assign the minimum available sample size to all slices
# this is done by defining a new column: which is isBalancedSample
# if you only like to do partial balancing on some slice Values
# specify those slice values in sliceCombinValues_toBalance
def BalanceSampleSize(
    df, sliceCols, itemCols=None, sliceCombinValues_toBalance=None):

  df2 = df.copy()
  if itemCols is None:
    itemCols = ['dummy_item']
    df2['dummy_item'] = range(len(df2))

  df2 = Concat_stringColsDf(
    df=df2.copy(),
    cols=itemCols,
    colName='item_combin',
    sepStr='-')

  df2 = Concat_stringColsDf(
    df=df2.copy(),
    cols=sliceCols,
    colName='slice_combin',
    sepStr='-')

  itemColsStr = '_'.join(itemCols)
  sliceColsStr = '_'.join(sliceCols)

  df3 = (df2[['item_combin', 'slice_combin']].
         drop_duplicates().sort_values('slice_combin').reset_index(drop=True))

  dfItemCount_perSlice = df3.groupby(['slice_combin'], as_index=False).agg(len)
  dfItemCount_perSlice.rename(
      columns={'item_combin':'item_combin_count'}, inplace=True)
  dfItemCount_perSlice['slice_item_index'] =  dfItemCount_perSlice[
    'item_combin_count'].map(range)

  if sliceCombinValues_toBalance is None:
    minSs = dfItemCount_perSlice['item_combin_count'].min()
    # if there is only once slice remaining, we assign False to all
    if len(dfItemCount_perSlice) < 2:
      minSs = -float('inf')
  else:
    df0 = dfItemCount_perSlice[dfItemCount_perSlice['slice_combin'].isin(
        sliceCombinValues_toBalance)]
    minSs =  df0['item_combin_count'].min()
    # if there is only once slice remaining, we assign False to all
    if len(df0) < 2:
      minSs = -float('inf')

  dfItemSliceIndex = Flatten_RepField(
      df=dfItemCount_perSlice, listCol='slice_item_index')
  del dfItemSliceIndex['item_combin_count']

  colName = sliceColsStr + '.' + itemColsStr + '_index'
  boolColName = 'balanced_' + sliceColsStr + '__' + itemColsStr
  dfItemSliceIndex.rename(
      columns={'slice_item_index': colName},
      inplace=True)

  if sliceCombinValues_toBalance is None:
    dfItemSliceIndex[boolColName] = dfItemSliceIndex[colName] < minSs
  else:
    dfItemSliceIndex[boolColName] = (
        (dfItemSliceIndex[colName] < minSs) |
        ~dfItemSliceIndex['slice_combin'].isin(sliceCombinValues_toBalance))

  df3[colName] = dfItemSliceIndex[colName]
  df3[boolColName] = dfItemSliceIndex[boolColName]
  fullDf = pd.merge(df2, df3, how='left', on=['item_combin', 'slice_combin'])

  df0 = fullDf[sliceCols + [boolColName, 'item_combin']].copy()
  g = df0.groupby(sliceCols + [boolColName], as_index=False)
  infoDf = g.agg({'item_combin':{'_count':lambda x: len(set(x))}})
  infoDf.columns = [''.join(col).strip() for col in infoDf.columns.values]

  for col in ['item_combin', 'slice_combin']:
    del fullDf[col]

  subDf = fullDf[fullDf[boolColName]].reset_index(drop=True)
  for col in [colName, boolColName]:
    del subDf[col]

  return {'fullDf': fullDf, 'subDf':subDf, 'infoDf':infoDf}

'''
df = GenUsageDf_forTesting()
Mark(df[:2])

res = BalanceSampleSize(
    df=df,
    sliceCols=['country'],
    itemCols=['user_id', 'date'],
    sliceCombinValues_toBalance=None)
Mark(res['infoDf'])

## partial balancing
res = BalanceSampleSize(
    df=df,
    sliceCols=['country'],
    itemCols=['user_id', 'date'],
    sliceCombinValues_toBalance=['JP', 'FR'])

Mark(res['infoDf'])
'''

## This will make sure that the sample size is the same
# for each (multi-dimensional) value of "wrtCols" across
# sliceCols. For example for if wrtCols = [country], sliceCols = [expt_id]
# for Japan we will have same number of
# units on base and test arms eg 2 and 2
# and for US we will have same number eg 3 and 3.
# TODO: Reza Hosseini resolve BUG: if RU has 3 items on base and no items on
# test. RU base will be kept at 3. Maybe RU has to be dropped.
def BalanceSampleSize_wrtCols(
    df,
    sliceCols,
    wrtCols,
    itemCols=None,
    sliceCombinValues_toBalance=None):

  df2 = df.copy()
  df2 = Concat_stringColsDf(
    df=df2.copy(),
    cols=wrtCols,
    colName='wrt_combin',
    sepStr='-')

  g = df2.groupby(['wrt_combin'], as_index=False)

  def F(group):
    res = BalanceSampleSize(
      df=group,
      sliceCols=sliceCols,
      itemCols=itemCols,
      sliceCombinValues_toBalance=sliceCombinValues_toBalance)

    return res['fullDf']

  fullDf = g.apply(F)

  itemColsStr = '_'.join(itemCols)
  sliceColsStr = '_'.join(sliceCols)
  boolColName = 'balanced_' + sliceColsStr + '__' + itemColsStr

  df0 = fullDf[sliceCols + wrtCols + itemCols + [boolColName]].copy()
  df0 = Concat_stringColsDf(
    df=df0.copy(),
    cols=itemCols,
    colName='item_combin',
    sepStr='-')

  g = df0.groupby(sliceCols + wrtCols + [boolColName], as_index=False)
  infoDf = g.agg({'item_combin':{'_count':lambda x: len(set(x))}})
  infoDf.columns = [''.join(col).strip() for col in infoDf.columns.values]

  subDf = fullDf[fullDf[boolColName]].reset_index(drop=True)

  return {"df":fullDf, "infoDf":infoDf}

'''
df = GenUsageDf_forTesting()
Mark(df[:2])

BalanceSampleSize_wrtCols(
    df=df,
    sliceCols=['expt'],
    wrtCols=['country'],
    itemCols=['user_id'],
    sliceCombinValues_toBalance=None)
'''

## counts items in each slice
# it then tests if the difference is sig
def TestObsCounts_acrossSlices(df, countCol, sliceCols, popSize):

  df0 = df[[countCol] + sliceCols].drop_duplicates()
  countDf = df0.groupby(sliceCols, as_index=False).agg(lambda x: len(set(x)))
  from statsmodels.stats.proportion import proportions_ztest
  counts = np.array(countDf[countCol].values)
  nobs = np.array([popSize] * len(countDf))
  stat, pval = proportions_ztest(counts, nobs)
  print('{0:0.3f}'.format(pval))

  return {'countDf':countDf, 'pval':pval}

'''
df = GenUsageDf_forTesting()
TestObsCounts_acrossSlices(df=df, countCol='user_id', sliceCols=['expt'], popSize=100)
'''

## calculate # users, # (user, dates), user per day
def UserPerDate_simple(
    df,
    usageCols,
    userCol='gaia_id',
    dateCol='date',
    compareCol='expt',
    topNum=10,
    condDictPre=None,
    regDictPre=None,
    condDictPost=None,
    regDictPost=None,
    sizeAlpha=0.75):

  df0 = df.copy()

  if len(usageCols) > 1:
    df0 = Concat_stringColsDf(df0, cols=usageCols, colName=None, sepStr='-')
  usageCol = '-'.join(usageCols)

  condStr = DictOfLists_toString(
      condDictPre,
      dictElemSepr='__',
      listElemSepr=',',
      keyValueSepr=':')

  if condDictPre is not None:
    ind = BuildCondInd(df=df0, condDict=condDictPre)
    df0 = df0[ind].copy()
    df0 = df0.reset_index(drop=True)

  topUsages = list(df0[usageCol].value_counts().keys()[:topNum])
  chosenUsages = topUsages

  gbCols = [compareCol, usageCol]
  countCols = [userCol]
  df1 = df0[gbCols + countCols]
  g = df1.groupby(gbCols, as_index=False)
  dfAgg1 = g.aggregate({userCol:{'_num':lambda x: len(set(x))}})
  dfAgg1.columns = [''.join(col).strip() for col in dfAgg1.columns.values]
  dfAgg1 = dfAgg1[dfAgg1[usageCol].isin(chosenUsages)]

  if regDictPost is not None:
    dfAgg1 = dfAgg1.reset_index(drop=True)
    dfAgg1 = dfAgg1[BuildRegexInd(df=dfAgg1, regDict=regDictPost)]
    dfAgg1 = dfAgg1.reset_index(drop=True)

  res = PivotPlotWrt(
      df=dfAgg1,
      pivotIndCol=usageCol,
      compareCol=compareCol,
      valueCol=userCol + '_num',
      pltTitle='num of distinct users with usage: ' + condStr,
      sizeAlpha=sizeAlpha)

  itemCols = [userCol, dateCol]
  df0 = Concat_stringColsDf(
      df=df0,
      cols=itemCols,
      colName='item',
      sepStr='-')

  df0 = df0[['item'] + [compareCol] + [usageCol]]
  g = df0.groupby(['expt'] + [usageCol], as_index=False)
  dfAgg2 = g.agg({'item': {'_with_usage_count': lambda x: len(set(x))}})
  dfAgg2.columns = [''.join(col).strip() for col in dfAgg2.columns.values]
  dfAgg2 = dfAgg2.sort_values(['item_with_usage_count'], ascending=[0])
  dfAgg2 = dfAgg2[dfAgg2[usageCol].isin(chosenUsages)]

  if regDictPost is not None:
    dfAgg2 = dfAgg2.reset_index(drop=True)
    dfAgg2 = dfAgg2.sort_values([usageCol, compareCol]).reset_index(drop=True)
    dfAgg2 = dfAgg2[BuildRegexInd(df=dfAgg2, regDict=regDictPost)]
    dfAgg2 = dfAgg2.reset_index(drop=True)

  res = PivotPlotWrt(
      df=dfAgg2,
      pivotIndCol=usageCol,
      compareCol=compareCol,
      valueCol='item_with_usage_count',
      pltTitle='num of (user, date) with usage: ' + condStr,
      sizeAlpha=sizeAlpha)

  p = res['plt']['fig']

  dfAgg = pd.merge(dfAgg1, dfAgg2, on=[compareCol, usageCol])
  dfAgg['num_days_with_usage_per_user'] = (
      dfAgg['item_with_usage_count'] /
      dfAgg['gaia_id_num'])

  if regDictPost is not None:
    dfAgg = dfAgg.reset_index(drop=True)
    dfAgg = dfAgg.sort_values([usageCol, compareCol]).reset_index(drop=True)
    dfAgg = dfAgg[BuildRegexInd(df=dfAgg, regDict=regDictPost)]
    dfAgg = dfAgg.reset_index(drop=True)

  res = PivotPlotWrt(
    df=dfAgg,
    pivotIndCol=usageCol,
    compareCol=compareCol,
    valueCol='num_days_with_usage_per_user',
    pltTitle='num of days with usage per user: ' + condStr,
    sizeAlpha=sizeAlpha)

  return dfAgg
