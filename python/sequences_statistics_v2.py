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

""" Statistics on sequences: new approach which allows to
calculate the metrics for various slices
we develop a method in which first
we calculate summable metrics for sequential data
the summable data will be augmented with appropriate totals
which are also summable
for example a seqCol can be mailingFeat-comp > mailingFeat-phone > search-phone
Assume we want to be able to slice by sliceCols=[country, date]
we also like to keep some seqPropCols
(columns which keep track of the sequence property) of interest around
for example seqPropCols=[form_factor_mix]
which tells us if there was an an interface (comp/pc) change in the seq
then consider the data country, date, device, sequence, count,
event_1st, event_2nd, event_3rd, event_4th
the input seqDecompCols =
[event_1st=a1, event_2nd=a2, event_3rd=a3, event_4th=a4]
keeps tracks of the decompositions

we first add summable columns as follows:
sliceCols [eg country, date] = (c, date) + seqCol [sequence]=s + count + ...
new column: sequence=num of s occurrences for the sliceCols=(c, date)
new column:  num of ith_event occurrences=ai

This intermediate data set will can be used to calculate probabilities
for slices of interest
this includes calculating:
P(S = s | sliceCols)
P(Si=si | sliceCols)  Si is the ith element of the sequence
P(S=s | sliceCols) / PRODUCT(P(Si=si | sliceCols)), i = 1, 2, 3, k
where k=2,3,4 depending on what length of sequences.
the number k should be specified by the second function.

stage 1: creates a df with event combinations and writes to disk
count distinctCols are to count number of sequences

TODO remove trim from here, do trim=4
"""

def CreateSeqTab_addEventCombin(
    df,
    seqDimCols,
    partitionCols0,
    sliceCols,
    seqPropCols,
    timeGap,
    timeCol,
    trim,
    countDistinctCols,
    seqCol="trimmed_seq_deduped",
    noShifted=False,
    timeColEnd=None,
    addSeqPropMixedCols=False,
    ordered=True,
    fn=None,
    writePath=''):

  partitionCols = partitionCols0 + sliceCols
  extraCols = seqPropCols

  ## we need to insure that seqDimCols determine seqPropCols
  # otherwise, sequence_count and sequence_count_agg do not agree
  # however this could be true without: set(seqPropCols) <= set(seqDimCols)
  # while the subset condition is not needed to satisfy
  # therefore we remove this error below
  # we will just give a warning
  '''
  assert (set(seqPropCols) <= set(seqDimCols)),("seqPropCols must be a subset of seqDimCols" +
    " otherwise, sequence_count and sequence_count_agg do not agree" +
    " therefore calculated prob are unreliable")
  '''

  if not set(seqPropCols) <= set(seqDimCols):
    warnings.warn(
        "\n *** WARNING: seqPropCols should be" +
        "typically a subset of seqDimCols." +
        " If this is not the case, make sure seqDimCols determines" +
        " seqPropCols uniquely. \n")

  seqDf = CreateSeqDf(
      df=df,
      timeCol=timeCol,
      seqDimCols=seqDimCols,
      partitionCols=partitionCols,
      timeGap=timeGap,
      trim=trim,
      timeColEnd=timeColEnd,
      extraCols=extraCols,
      ordered=ordered)

  if noShifted:
    seqDf = seqDf[seqDf["seq_shift_order"].map(str) == "0"]

  print(seqDf.columns)

  eventCols = ['event_1', 'event_2', 'event_3', 'event_4'][:trim]

  # if seqCol is the standard column, the seq is simply
  # the concat of the events
  # otherwise we need to extract the events again
  if seqCol != "trimmed_seq_deduped":
    for col in ['event_1', 'event_2', 'event_3', 'event_4']:
      del seqDf[col]
    seqDf = AddSeqOrdEvent(
        df=seqDf.copy(),
        seqCol=seqCol,
        sepStr='>',
        noneValue='BLANK')

  seqDf['seq'] = seqDf['event_1']
  for i in range(1, trim):
    seqDf['seq'] = seqDf['seq'] + '>' + seqDf[eventCols[i]]

  ## adding the seq property, mixture columns
  if addSeqPropMixedCols and len(seqPropCols) > 0:
    seqPropCols_mix = ['trimmed_' + x + '_parallel_mix' for x in seqPropCols]
    seqPropCols = (
        ['trimmed_' + x + '_parallel' for x in seqPropCols]
        + seqPropCols_mix)

  ## we only need to keep unique combinations involving countDistinctCols
  keepCols = (
      countDistinctCols + sliceCols + seqPropCols + eventCols + ['seq'])
  seqDf = seqDf[keepCols].copy()
  seqDf  = seqDf.drop_duplicates()

  eventCombinCols = []
  for i in range(1, trim+1):
    for j in range(i+1, trim+1):
      col = 'event_' + str(i) + '_to_' + str(j)
      eventCombinCols.append(col)
      seqDf[col] = seqDf['event_' + str(i)]
      for k in range(i+1, j+1):
        #print(range(i+1, j+1))
        seqDf[col] = seqDf[col] + '>' + seqDf['event_' + str(k)]
  ## lets first aggregate to the crudest needed level

  '''
  seqDfWithTotals = CalcSeqCountWrtId(
      seqDf=seqDf,
      sliceCols=sliceCols,
      seqPropCols=seqPropCols,
      seqCol='seq',
      seqCountCol='seq_count',
      seqLenCol='seq_length')
  '''
  print(seqDf[:5])

  if fn != None:
    WriteCsv(df=seqDf, fn=writePath + fn + '.csv', printLog=True)

  #out = {'seqDf': seqDf, 'seqDfWithTotals': seqDfWithTotals}
  return seqDf

'''
df = GenUsageData(
    userNum=2,
    dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
    dt2=datetime.datetime(2017, 4, 12, 20, 0, 0))
df['date'] = df['time'].dt.date

out = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date'],
    seqPropCols=['form_factor'],
    timeGap=1*60,
    timeCol='time',
    trim=4,
    timeColEnd='end_time',
    countDistinctCols=['seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

'''

## takes a seq data frame, and calculates counts wrt an index
# it calculate the metrics per slice given in partitionCols
# therefore one can aggregate further if needed
# (and if summable i.e. no index overlap in slices)
# sliceCols are the columns for which the totals are calculated over them:
# examples for sliceCols:
# country, local_time_range for apps, second one is seqPropCol typically
# auxSliceCols are not considered when adding totals: example party_mix for apps
# therefore the seqCol should determine the auxSliceCol values (uniquely)
def CalcSeqCountWrtId(
    seqDf,
    trim,
    sliceCols=[],
    auxSliceCols=[],
    seqCol='seq',
    countDistinctCols=['seq_id']):

  for col in auxSliceCols:
    df0 = seqDf[[seqCol, col]].copy()
    df0 = df0.reset_index(drop=True)
    df0 = df0.drop_duplicates()
    df0 = df0.reset_index(drop=True)
    #assert (len(set(seqDf[seqCol].values)) == len(df0)),(
    #    col + " is not determined uniquely by seqCol (seqDimCols).")
    if (len(set(seqDf[seqCol].values)) != len(df0)):
      warnings.warn(col + " is not determined uniquely by seqCol (seqDimCols).")

  seqDf = Concat_stringColsDf(
      df=seqDf, cols=countDistinctCols, colName='distinct_id', sepStr='-')

  seqDecompCols = ['event_1', 'event_2', 'event_3', 'event_4'][:trim]
  eventCombinCols = []
  for i in range(1, trim+1):
    for j in range(i+1, trim+1):
      col = 'event_' + str(i) + '_to_' + str(j)
      eventCombinCols.append(col)

  gbCols = sliceCols + seqDecompCols + auxSliceCols + eventCombinCols + [seqCol]
  allCols = gbCols + ['distinct_id']
  seqDf2 = seqDf[allCols]

  g = seqDf2.groupby(gbCols)
  seqDf2 = g.agg({'distinct_id': lambda x: len(set(x))}, as_index=False)
  seqDf2 = seqDf2.reset_index()
  seqDf2 = seqDf2.rename(columns={'distinct_id': 'count'})

  seqDfWithTotals = AddTotalsDf(
      df=seqDf2,
      categCols=[seqCol] + seqDecompCols + eventCombinCols,
      valueCols=['count'],
      sliceCols=sliceCols,
      aggFnDict=sum,
      integOutOther=False)

  ## this is sanity checking to make sure the totals are calculated correctly
  # wrt the right slicing
  # this is a looser condition than the one with warnings on top
  seqDfWithTotals2 = seqDfWithTotals[[seqCol] + sliceCols].copy()
  seqDfWithTotals2 = seqDfWithTotals2.drop_duplicates()

  if (len(seqDfWithTotals) != len(seqDfWithTotals2)):
    Mark(seqDfWithTotals[:50], 'seqDfWithTotals')
    Mark(seqDfWithTotals2[:50], 'seqDfWithTotals')

  assert (len(seqDfWithTotals) == len(seqDfWithTotals2)),(
      "the combination of seq column and sliceCols," +
      " does not uniquely determine the auxSliceCols." +
      " This will cause errors in calculating probabilities," +
      " since the total count per slice will not be correct.")

  return(seqDfWithTotals)


'''
df = GenUsageData(
    userNum=2,
    dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
    dt2=datetime.datetime(2017, 4, 12, 20, 0, 0))
df['date'] = df['time'].dt.date

seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date'],
    seqPropCols=['form_factor'],
    timeGap=1*60,
    timeCol='time',
    trim=3,
    timeColEnd='end_time',
    countDistinctCols=['seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

seqCountDf = CalcSeqCountWrtId(
    seqDf=seqDf,
    trim=3,
    sliceCols=['date'],
    auxSliceCols=map(lambda x: x + '_parallel', ['form_factor']) + ['seq_len'],
    seqCol='seq',
    countDistinctCols = ['seq_id'])
'''

## it takes a seq df with counts and calculates
# Relative probability 1: P(S1, ..., Sk) / P(S1)*...*P(Sk)
# Relative probability 2: P(Sk  |  Sk-1, ..., S1) / P(Sk | Sk-1, ..., S2)
# as well as confidence intervals
# this function does not keep track of any slicing
# or does not agg the data before calculating metrics
# and that is by design, since this fcn only adds new cols from old cols
# which is known as composite metrics
def AddSeqProbCiMetrics(
    seqDf,
    trim,
    addCounts=True,
    sliceCols=[],
    auxSliceCols=[],
    seqCol='seq',
    countDistinctCols=['seq_id'],
    seqCountMin=10):

  ## we add count cols if the data is not count data already
  if addCounts:
    df = CalcSeqCountWrtId(
        seqDf,
        trim=trim,
        sliceCols=sliceCols,
        auxSliceCols=auxSliceCols,
        seqCol=seqCol,
        countDistinctCols=countDistinctCols)
  else:
    df = seqDf

  ## this determines the length of the seq
  seqDecompCols = ['event_1', 'event_2', 'event_3', 'event_4'][:trim]
  eventCombinCols = []
  for i in range(1, trim+1):
    for j in range(i+1, trim+1):
      col = 'event_' + str(i) + '_to_' + str(j)
      eventCombinCols.append(col)

  valueCols_prefix = [seqCol] + seqDecompCols + eventCombinCols
  df['estimator_var1'] = 0
  df['estimator_var2'] = 0
  df['prob_prod'] = 1
  df[seqCol + '_prob'] = (1.0 * df[seqCol + '_count' + '_agg'] /
                          df['count_slice_total'])
  ## calculate var probabilities and variances
  ## the total variance is also calculated since its the sum of all
  for valueCol0 in valueCols_prefix:
    x = (1.0 * df[valueCol0 + '_count' + '_agg'] /
         df['count_slice_total'])
    df[valueCol0 + '_var_comp'] = (1.0 / (df[valueCol0 + '_count' + '_agg']) +
      (1.0 - x) / (1.0 * x * df['count_slice_total']))
  ## calculate the prod probability for the seq elements (seqDecompCols)
  for valueCol0 in seqDecompCols:
    df['prob_prod'] =  df['prob_prod'] * (1.0 * df[valueCol0 + '_count' + '_agg'] /
                               df['count_slice_total'])
  ## variance for method 1
  for valueCol0 in ([seqCol] + seqDecompCols):
    df['estimator_var1'] = (df['estimator_var1'] +
                           1.0 * df[valueCol0 + '_var_comp'])
  ## calculate the relative:  P(X1=a1, ..., Xn=an) / Prod P(Xi=ai)
  df['relative_prob1'] = 1.0 * df[seqCol + '_prob'] / df['prob_prod']
  ## calculating the second metric:  R2(s) = P[1, k]P[2, k-1] / P[1, k-1]P[2, k]
  # variables we need  event_[1, k], event_[2, k-1], event_[1, k-1], event_[2, k]
  e_1_to_k = 'event_1_to_' + str(trim)
  e_2_to_k_minus_1 = 'event_2_to_' + str(trim - 1)
  e_1_to_k_minus_1 = 'event_1_to_' + str(trim - 1)
  e_2_to_k = 'event_2_to_' + str(trim)

  ## reset values for trim=2
  if trim == 2:
    e_2_to_k_minus_1 = 'event_2'
    e_1_to_k_minus_1 = 'event_1'
  ## reset values for trim=3
  if trim == 3:
    e_2_to_k_minus_1 = 'event_2'

  newValueCols0 = [e_1_to_k, e_2_to_k_minus_1, e_1_to_k_minus_1, e_2_to_k]

  ## here we calculate the relative metric type 2: R2(s) = P[1, k]P[2, k-1] / P[1, k-1]P[2, k]
  if trim == 2:
    numer = (1.0 * df[e_1_to_k + '_count' + '_agg'] /
                               df['count_slice_total'])
    denom = (1.0 * df['event_1' + '_count' + '_agg'] * df['event_2' + '_count' + '_agg'] /
                               (df['count_slice_total']**2))
    df['relative_prob2'] = numer / denom
    df['estimator_var2'] = (1.0 * df[e_1_to_k + '_var_comp'] +
                            df[e_1_to_k_minus_1 + '_var_comp'] +
                            df[e_2_to_k_minus_1 + '_var_comp'])

  else:
    numer = 1.0 * df[e_1_to_k + '_count' + '_agg'] *  df[e_2_to_k_minus_1 + '_count' + '_agg']
    denom = 1.0 * df[e_1_to_k_minus_1 + '_count' + '_agg'] * df[e_2_to_k + '_count' + '_agg']

    df['relative_prob2'] = 1.0 * numer / denom

    df['estimator_var2'] = (1.0 * df[e_1_to_k + '_var_comp'] +
                            df[e_2_to_k_minus_1 + '_var_comp'] +
                            df[e_1_to_k_minus_1 + '_var_comp'] +
                            df[e_2_to_k_minus_1 + '_var_comp'])

  def AddConfInt(df, probCol, varCol, colSuffix):
    df2 = df.copy()
    df['CI_width_log_relative_prob'] = 1.96 * df[varCol].map(math.sqrt)
    df['log_relative_prob_lower'] = (df[probCol].map(math.log) -
                                     df['CI_width_log_relative_prob'])
    df['log_relative_prob_upper'] = (df[probCol].map(math.log) +
                                     df['CI_width_log_relative_prob'])
    ## go back to original scale and round
    df['relative_prob_lower'] = (
        df['log_relative_prob_lower'].map(math.exp).apply(lambda x: round(x, 3)))
    df['relative_prob_upper'] = (
        df['log_relative_prob_upper'].map(math.exp).apply(lambda x: round(x, 3)))
    ## small num of seq should not be significant, we reset values
    df['relative_prob_lower'][df['count'] < seqCountMin] = 0
    df['relative_prob_upper'][df['count'] < seqCountMin] = (
        df[probCol][df['count'] < seqCountMin])*2 + 10
    df2['relative_prob_lower' + colSuffix] = df['relative_prob_lower']
    df2['relative_prob_upper' + colSuffix] = df['relative_prob_upper']

    return df2

  df = AddConfInt(
      df=df.copy(), probCol='relative_prob1',
      varCol='estimator_var1', colSuffix='1')
  df = AddConfInt(
      df=df.copy(), probCol='relative_prob2',
      varCol='estimator_var2', colSuffix='2')

  ## upper and lower bounds for the probability
  # we use a conservative estimate of the variance p(1-p)
  # we use p=1/2
  df['prob_var'] = (
      (1.0*df['seq_prob']*(1-df['seq_prob']) / df['count_slice_total']))
      #+ 1.0/(16.0*df['count_slice_total']))
  df['seq_prob_upper'] = (
      df['seq_prob'] +
      1.96 *(df['prob_var']).map(math.sqrt)).map(lambda x: min(x, 1.0))
  df['seq_prob_lower'] = (
      df['seq_prob'] -
      1.96 * (df['prob_var']).map(math.sqrt)).map(lambda x: max(x, 0.0))

  return df


'''
df = Sim_depUsageData()

seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date'],
    seqPropCols=['form_factor'],
    timeGap=3*60,
    timeCol='time',
    trim=3,
    timeColEnd='end_time',
    countDistinctCols=['seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

seqCountDf = CalcSeqCountWrtId(
    seqDf=seqDf,
    trim=3,
    sliceCols=['date'],
    auxSliceCols=map(lambda x: x + '_parallel', ['form_factor']) + ['seq_len'],
    seqCol='seq',
    countDistinctCols = ['seq_id'])

seqDfWithSignif = AddSeqProbCiMetrics(
    seqDf=seqDf,
    trim=3,
    sliceCols=[],
    auxSliceCols=[],
    seqCol='seq',
    countDistinctCols=['seq_id'])

Mark(seqCountDf.sort_values(['count'], ascending=[0])[:5])
Mark(seqDfWithSignif.sort_values(['count'], ascending=[0])[:5])
'''

'''
## Example 2
trim = 3
df = Sim_depUsageData(userNum=100, subSeqLen=3, repeatPattern=None)

seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date', 'country'],
    seqPropCols=['form_factor', 'start_hour'],
    timeGap=3*60,
    timeCol='time',
    trim=trim,
    timeColEnd='end_time',
    countDistinctCols=['user_id', 'seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

#auxSliceCols = (
#    map(lambda x: 'trimmed_' + x + '_parallel', seqPropCols) +
#    map(lambda x: 'trimmed_' + x + '_parallel' + '_mix', seqPropCols) +
#    ['seq_len'])

sliceCols = [
    #'date',
    'country'
    #'trimmed_start_hour_parallel',
    #'trimmed_start_hour_parallel_mix',
    #'trimmed_form_factor_parallel',
    #'trimmed_form_factor_parallel_mix']

auxSliceCols = [
    #'trimmed_form_factor_parallel',
    #'trimmed_form_factor_parallel_mix']


## calc sig
seqDfWithSignif = AddSeqProbCiMetrics(
    seqDf=seqDf,
    trim=trim,
    sliceCols=sliceCols,
    auxSliceCols=auxSliceCols,
    seqCol='seq',
    countDistinctCols=['user_id'],
    seqCountMin=5)

Mark(seqDfWithSignif.sort_values(['count'], ascending=[0])[:10])

Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif,
    sliceCols=sliceCols,
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    seqNumLimit=None,
    rotation=0,
    figSize=[5, 20])

'''

## stage 2:
# calc prob metrics from summable cols
# only apply this is the columns are summable
# for this to be true we need:
# sliceColsAgg would not have counted the same countWrtId
# therefore by aggregating (summing counts) further to
# sliceColsAgg the counts are valid
def CalcProbMetrics_fromSummable(
    seqCountDf,
    sliceColsAgg,
    trim,
    auxSliceCols=[],
    seqCountMin=10,
    removeBlankSeqs=True,
    addSeqPropMixedCols=False):

  seqDecompCols = ['event_1', 'event_2', 'event_3', 'event_4'][:trim]
  eventCombinCols = []
  for i in range(1, trim+1):
    for j in range(i+1, trim+1):
      col = 'event_' + str(i) + '_to_' + str(j)
      eventCombinCols.append(col)

  valueCols = (
      ['count'] +
      [x + '_count_agg' for x in seqDecompCols] +
      [x + '_count_agg' for x in eventCombinCols] +
      ['count_slice_total','seq_count_agg'])

  mainIndCols = ['seq'] + seqDecompCols

  seqDfWithTotals2 = seqCountDf[
      mainIndCols + auxSliceCols + sliceColsAgg + valueCols]
  g = seqDfWithTotals2.groupby(mainIndCols + auxSliceCols + sliceColsAgg)
  seqDfWithTotals3 = g.agg(sum)
  seqDfWithTotals3 = seqDfWithTotals3.reset_index()

  seqDfWithSignif = AddSeqProbCiMetrics(
      seqDf=seqDfWithTotals3,
      trim=trim,
      addCounts=False,
      seqCountMin=seqCountMin)
  seqDfWithSignif = seqDfWithSignif.sort_values(['seq'] + sliceColsAgg)

  return seqDfWithSignif

'''
trim = 3
df = Sim_depUsageData(
    userNum=5, subSeqLen=3, repeatPattern=200)

seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date', 'country'],
    seqPropCols=['form_factor'],
    timeGap=3*60,
    timeCol='time',
    trim=trim,
    timeColEnd='end_time',
    countDistinctCols=['seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

auxSliceCols = (
    map(lambda x: 'trimmed_' + x + '_parallel', seqPropCols) +
    map(lambda x: 'trimmed_' + x + '_parallel' + '_mix', seqPropCols) +
    ['seq_len'])

seqCountDf = CalcSeqCountWrtId(
    seqDf=seqDf,
    trim=trim,
    sliceCols=['date', 'country'],
    auxSliceCols=auxSliceCols,
    seqCol='seq',
    countDistinctCols = ['seq_id'])

## indirect method: step 1
seqDfWithSignif1 = AddSeqProbCiMetrics(
    seqDf=seqDf,
    trim=trim,
    sliceCols=['date', 'country'],
    auxSliceCols=auxSliceCols,
    seqCol='seq',
    countDistinctCols=['seq_id'],
    seqCountMin=5)

## step 2
seqDfWithSignif2 = CalcProbMetrics_fromSummable(
    seqCountDf=seqCountDf,
    sliceColsAgg=['country'],
    trim=trim,
    auxSliceCols=auxSliceCols,
    seqCountMin=5,
    removeBlankSeqs=True)

Mark(seqDfWithSignif2.sort_values(['count'], ascending=[0])[:10])

Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif1,
    sliceCols=['country'],
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    seqNumLimit=None,
    rotation=0,
    figSize=[5, 20])

map(Mark, range(10))
Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif2,
    sliceCols=['country'],
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    seqNumLimit=None,
    rotation=0,
    figSize=[5, 20])
'''

## calculates sig sequences and also adds penetration CI for the seq
def FindSigSeq_withPenet(
    seqDf,
    trim,
    seqCol,
    sliceCols,
    auxSliceCols,
    countDistinctCols,
    penetItemCols,
    seqCountMin=5):

  ## calc sig
  seqDfWithSignif = AddSeqProbCiMetrics(
      seqDf=seqDf.copy(),
      trim=trim,
      sliceCols=sliceCols,
      auxSliceCols=auxSliceCols,
      seqCol=seqCol,
      countDistinctCols=countDistinctCols,
      seqCountMin=seqCountMin)

  #Mark(seqDfWithSignif.sort_values(['count'], ascending=[0])[:10])
  if penetItemCols is None:
    penetItemCols = countDistinctCols

  ## adding penetrations
  seqDf2 = seqDf.copy()
  seqDf2 = seqDf2[penetItemCols + sliceCols + auxSliceCols + [seqCol]]

  usageCols = [seqCol]

  penetDf = CalcItemPenet(
      df=seqDf2,
      itemCols=penetItemCols,
      sliceCols=sliceCols,
      usageCols=usageCols)

  seqDfWithSignif2 = pd.merge(
      seqDfWithSignif, penetDf, on=sliceCols + usageCols)

  return seqDfWithSignif2

'''

trim = 3
df = Sim_depUsageData(userNum=100, subSeqLen=3, repeatPattern=None)

seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date', 'country'],
    seqPropCols=['form_factor', 'start_hour'],
    timeGap=3*60,
    timeCol='time',
    trim=trim,
    timeColEnd='end_time',
    countDistinctCols=['user_id', 'seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

#auxSliceCols = (
#    map(lambda x: 'trimmed_' + x + '_parallel', seqPropCols) +
#    map(lambda x: 'trimmed_' + x + '_parallel' + '_mix', seqPropCols) +
#    ['seq_len'])

sliceCols = [
    #'date',
    'country'
    #'trimmed_start_hour_parallel',
    #'trimmed_start_hour_parallel_mix',
    #'trimmed_form_factor_parallel',
    #'trimmed_form_factor_parallel_mix']

auxSliceCols = [
    #'trimmed_form_factor_parallel',
    #'trimmed_form_factor_parallel_mix']

seqCol = 'seq'
countDistinctCols = ['user_id']

seqDfWithSignif2 = FindSigSeq_withPenet(
    seqDf=seqDf,
    trim=trim,
    seqCol='seq',
    sliceCols=sliceCols,
    auxSliceCols=auxSliceCols,
    countDistinctCols='seq_id',
    penetItemCols=None,
    seqCountMin=5)

'''

## plotting the CI from sig sequences
def PlotSigSeq_concatSeqAndSlices(
    seqDfWithSignif,
    metricCol,
    metricColLower,
    metricColUpper,
    sliceCols=[],
    orderCol='count',
    removeBlankSeqs=True,
    rotation=75,
    figSize=[30, 3]):

  df = seqDfWithSignif.sort_values(['count'], ascending=[0])

  df = df[
      ['seq'] +
      sliceCols +
      [metricCol, metricColLower, metricColUpper]]

  df = Concat_stringColsDf(
      df,
      cols=['seq'] + sliceCols,
      colName='seq_slice',
      sepStr='-')

  ## subset the cases where at least one of the methods is sig
  df2 = df[df[metricColLower] > 1.1]

  if removeBlankSeqs:
    df2 = df2[df2['seq'].map(lambda x: 'BLANK' not in x)]

  n = len(df2)
  fig, ax = plt.subplots()
  fig.set_size_inches(figSize[0], figSize[1])
  plt.scatter(
      [(float(x) + 0.0) for x in range(n)],
      df2[metricCol],
      color='blue',
      alpha=0.8)

  plt.bar(
      left=[(float(x) + 0.0) for x in range(n)],
      height=df2[metricColUpper].values,
      bottom=df2[metricColLower].values,
      color='blue',
      width=0.25, alpha=0.5)

  labels = [item.get_text() for item in ax.get_xticklabels()]
  labels = list(df2['seq_slice'].values)
  ax.set_xticklabels(labels)

  locs, labels = plt.xticks([(float(x) + 0.5) for x in range(n)], labels)
  plt.setp(labels, rotation=rotation, fontweight='bold')
  plt.plot(range((n+2)), [1]*(n+2), color='orange', alpha=0.5)
  plt.plot(range((n+2)), [0]*(n+2), color='grey', alpha=0.5)
  plt.yscale('log')

## plotting the CI from sig sequences
# this will put confidence intervals
# from slices side by side
def Plt_sigSeq_compareSlices(
    seqDfWithSignif,
    sliceCols=[],
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    regDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    addPenetPlots=True,
    seqNumLimit=None,
    rotation=0,
    logScale=True,
    figSize=[5, 20],
    saveFig=False,
    figPath='',
    figFnPrefix='fig',
    figFnExt='png',
    Open=OpenFile):

  penetCols = []
  if addPenetPlots:
    penetCols = ['penetration', 'penetration_ci_lower', 'penetration_ci_upper']

  df = seqDfWithSignif.sort_values(['count'], ascending=[0]).copy()

    ## conditions to pick what product
  if condDict != None:
    df = df[BuildCondInd(df=df, condDict=condDict)]
    df = df.reset_index()

  if regDict != None:
    df = df[BuildRegexInd(df=df, regDict=regDict)]
    df = df.reset_index()

  df = df[
      ['seq'] +
      sliceCols +
      ['count', metricCol, metricColLower, metricColUpper] +
      ['seq_prob', 'seq_prob_upper', 'seq_prob_lower'] +
      penetCols]

  df = Concat_stringColsDf(
      df=df,
      cols=sliceCols,
      colName='slice_comb',
      sepStr='-')
  df = df.reset_index()

  ## subset the cases where at least one of the methods is sig
  df2 = df[df[metricColLower] > relativeProbLowerLim]
  if removeBlankSeqs:
    df2 = df2[df2['seq'].map(lambda x: 'BLANK' not in x)]
  df2 = df2.reset_index(drop=True)

  if len(df2) == 0:
    print('\n\n *** No data satisfied the conditions,' +
          'functions returns \n\n')
    return {'df': None, 'plotDict': None}

  df2 = df2.sort_values([orderByCol], ascending=[1])
  if seqNumLimit != None:
    df2 = df2[:seqNumLimit]

  plotDict = {}
  p1 = PlotCIWrt(
      df=df2.copy(),
      colUpper=metricColUpper,
      colLower=metricColLower,
      sliceCols=['slice_comb'],
      labelCol='seq',
      col=metricCol,
      ciHeight=0.5,
      rotation=0,
      addVerLines=[1, 2, 5],
      logScale=True,
      lowerLim=0.5,
      pltTitle=metricCol,
      figSize=figSize)

  plotDict['relative_prob'] = p1
  if saveFig:
    fn0 = figPath + figFnPrefix + '_relative_prob2.' + figFnExt
    fn = Open(fn0, 'w')
    p1.savefig(fn, bbox_inches='tight')

  #plt.figure()
  df = df2.copy()
  for col in ['seq_prob', 'seq_prob_upper',
              'seq_prob_lower']:
    df[col] = 100.0 * df[col]
  max0 = df['seq_prob_upper'].max() * 1.5
  min0 = df['seq_prob_lower'].min() / 5.0
  addVerLines = [
      0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
  addVerLines = [u for u in addVerLines if u <= max0]
  addVerLines = [u for u in addVerLines if u >= min0]

  p2 = PlotCIWrt(
      df=df.copy(),
      colUpper='seq_prob_upper',
      colLower='seq_prob_lower',
      sliceCols=['slice_comb'],
      labelCol='seq',
      col='seq_prob',
      ciHeight=0.5,
      rotation=0,
      addVerLines=addVerLines,
      logScale=logScale,
      lowerLim=-0.1,
      pltTitle='prob (%)',
      figSize=figSize)

  plotDict['seq_prob'] = p2

  if saveFig:
    fn0 = figPath + figFnPrefix + '_seq_probability.' + figFnExt
    fn = Open(fn0, 'w')
    p2.savefig(fn, bbox_inches='tight')

  df = df2.copy()

  if addPenetPlots:
    for col in ['penetration', 'penetration_ci_upper',
                'penetration_ci_lower']:
      df[col] = df[col]
    #m1 = (df['penetration_upper'].max() / 5).round()
    #m2 = max([1, m1]) + 2
    #addVerLines = [0.01, 0.1, 1, 2, m2/2.0, m2]
    max0 = df['penetration_ci_upper'].max() * 1.5
    min0 = df['penetration_ci_lower'].min() / 5.0
    addVerLines = [
        0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    addVerLines = [u for u in addVerLines if u <= max0]
    addVerLines = [u for u in addVerLines if u >= min0]

    p3 = PlotCIWrt(
        df=df.copy(),
        colUpper='penetration_ci_upper',
        colLower='penetration_ci_lower',
        sliceCols=['slice_comb'],
        labelCol='seq',
        col='penetration',
        ciHeight=0.5,
        rotation=0,
        addVerLines=addVerLines,
        logScale=logScale,
        lowerLim=-0.1,
        pltTitle='penetration (%)',
        figSize=figSize)
    plotDict['penet'] = p3

    if saveFig:
      fn0 = figPath + figFnPrefix + '_user_penetration.' + figFnExt
      fn = Open(fn0, 'w')
      p3.savefig(fn, bbox_inches='tight')

  df2 = df.reset_index()
  return {'df': df2, 'plotDict': plotDict}


'''
trim = 2

df = Sim_depUsageData(userNum=5, subSeqLen=3, repeatPattern=200)

seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date', 'country'],
    seqPropCols=['form_factor'],
    timeGap=3*60,
    timeCol='time',
    trim=3,
    timeColEnd='end_time',
    countDistinctCols=['seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

seqCountDf = CalcSeqCountWrtId(
    seqDf=seqDf,
    trim=trim,
    sliceCols=['date', 'country'],
    auxSliceCols=map(lambda x: x + '_parallel', ['form_factor']) + ['seq_len'],
    seqCol='seq',
    countDistinctCols = ['seq_id'])

seqPropCols = ['form_factor']

auxSliceCols = (
    map(lambda x: x + '_parallel', seqPropCols) +
    map(lambda x: x + '_mix', seqPropCols) +
    ['seq_len'])

## indirect method: step 1
seqDfWithSignif1 = AddSeqProbCiMetrics(
    seqDf=seqDf,
    trim=trim,
    sliceCols=['date', 'country'],
    auxSliceCols=map(lambda x: x + '_parallel', ['form_factor']),
    seqCol='seq',
    countDistinctCols=['seq_id'],
    seqCountMin=5)

## step 2
seqDfWithSignif2 = CalcProbMetrics_fromSummable(
    seqCountDf=seqCountDf,
    sliceColsAgg=['country'],
    trim=trim,
    auxSliceCols=map(lambda x: x + '_parallel', ['form_factor']),
    seqCountMin=5,
    removeBlankSeqs=True)

out.sort_values(['count'], ascending=[0])[:10]

Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif1,
    sliceCols=['country'],
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    seqNumLimit=None,
    rotation=0,
    figSize=[5, 20])

map(Mark, range(10))

Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif2,
    sliceCols=['country'],
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    seqNumLimit=None,
    rotation=0,
    figSize=[5, 20])
'''

'''
## new example with penetration
trim = 3

## simulate data
df = Sim_depUsageData(userNum=100, subSeqLen=3, repeatPattern=None)

## simulate data
seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=['prod', 'form_factor'],
    partitionCols0=['user_id'],
    sliceCols=['date', 'country'],
    seqPropCols=['form_factor', 'start_hour'],
    timeGap=3*60,
    timeCol='time',
    trim=trim,
    timeColEnd='end_time',
    countDistinctCols=['user_id', 'seq_id'],
    addSeqPropMixedCols=True,
    ordered=True,
    fn=None,
    writePath='')

## calc sig and penetration
seqDf['seq_len'] = seqDf['seq'].map(lambda x: len(x.split('>')))

#auxSliceCols = (
#    map(lambda x: 'trimmed_' + x + '_parallel', seqPropCols) +
#    map(lambda x: 'trimmed_' + x + '_parallel' + '_mix', seqPropCols) +
#    ['seq_len'])

sliceCols = [
    #'date',
    'country'
    #'trimmed_start_hour_parallel',
    #'trimmed_start_hour_parallel_mix',
    #'trimmed_form_factor_parallel',
    #'trimmed_form_factor_parallel_mix']

auxSliceCols = [
    #'trimmed_form_factor_parallel',
    #'trimmed_form_factor_parallel_mix']

seqCol = 'seq'
countDistinctCols = ['user_id']

seqDfWithSignif2 = FindSigSeq_withPenet(
    seqDf=seqDf,
    trim=trim,
    seqCol='seq',
    sliceCols=sliceCols,
    auxSliceCols=auxSliceCols,
    countDistinctCols=countDistinctCols,
    penetItemCols=None,
    seqCountMin=5)

out = Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif2,
    sliceCols=sliceCols,
    metricCol='relative_prob2',
    metricColLower='relative_prob_lower2',
    metricColUpper='relative_prob_upper2',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower2',
    seqNumLimit=None,
    rotation=0,
    figSize=[5, 20])

'''

### end-to-end significant sequence table, and results

## this needs to be implemented depending on the sql lang you use
def DefineSqlTable_fromFile(fn, sqlTableName, fnSuff, filePath):
  """query to define a permanent table"""
  fileName = filePath + fn + fnSuff
  if fnSuff == ".csv":
    sqlStr = "<DefineSqlTable_fromFile needs to be implemented>" + fileName

  return sqlStr

def ExecSqlQuery(query):
  print("<ExecSqlQuery needs to be implemented>")
  return None


## creates appropriate filename for writing result
# also creates sql table name
# and the sql string for defining the same table name
def Gen_seqData_FnTableNames(
    dataNamesList,
    dataDesc,
    trim,
    timeGapDict,
    seqCol,
    countDistinctCols,
    writePath,
    DefineSqlTable_fromFile,
    condDict=None,
    fnSuff='.csv',
    sqlTablePrefix=""):

  timeGapStr = str(timeGapDict['value']) + timeGapDict['scale']

  condString = DictOfLists_toString(
      condDict, dictElemSepr='_', listElemSepr='_')
  dataString = '_'.join(dataNamesList)
  fn = dataString
  sqlTableName = sqlTablePrefix + '.' + dataString

  if dataDesc != '':
    fn = fn + '_' + dataDesc
    sqlTableName = sqlTableName + '.' + dataDesc

  if seqCol != '':
    fn = fn + '_' + seqCol
    sqlTableName = sqlTableName + '.' + seqCol

  if countDistinctCols not in [None, ""]:
    fn = fn + '_' + "_".join(countDistinctCols)
    sqlTableName = sqlTableName + '.' + "_".join(countDistinctCols)


  if condString != '':
    fn = fn + '_' + condString
    sqlTableName = sqlTableName + '.' + condString

  if trim != '':
    fn = fn + '_' + timeGapStr + '_len' + str(trim)
    sqlTableName = sqlTableName + '.' + timeGapStr + '_trim' + str(trim)

  sqlStr = DefineSqlTable_fromFile(
      fn=fn,
      sqlTableName=sqlTableName,
      fnSuff=fnSuff,
      filePath=writePath)

  return {'fn':fn, 'sqlTableName':sqlTableName, 'sqlStr':sqlStr}

## generate filename (fn) and sql table name
# for the data of interest with given trim
# and given writePath
# then it creates the file in appropriate place
# it also outputs the sql command
def WriteSeqTable_forSql(
    df,
    seqDimCols,
    partitionCols0,
    sliceCols,
    seqPropCols,
    timeGapMin,
    timeCol,
    timeColEnd,
    trim,
    countDistinctCols,
    condDict,
    addSeqPropMixedCols,
    ordered,
    writePath,
    dataNamesList,
    dataDesc,
    fnSuff,
    seqCol="trimmed_seq_deduped",
    noShifted=False,
    defineSqlTab=False,
    DefineSqlTable_fromFile=lambda: None,
    ExecSqlQuery=lambda: None,
    sqlTablePrefix="",
    timeGapDict=None,
    writeTableLogFn=None):

  if timeColEnd is None:
    timeColEnd = timeCol

  if timeGapDict is None:
    timeGapDict = {
        'scale': 'min',
        'value':timeGapMin,
        'to_seconds_converter':60}
  timeGap = timeGapDict['value'] * timeGapDict['to_seconds_converter']

  out = Gen_seqData_FnTableNames(
      dataNamesList=dataNamesList,
      dataDesc=dataDesc,
      trim=trim,
      timeGapDict=timeGapDict,
      seqCol=seqCol,
      countDistinctCols=countDistinctCols,
      condDict=condDict,
      writePath=writePath,
      fnSuff=fnSuff,
      sqlTablePrefix=sqlTablePrefix,
      DefineSqlTable_fromFile=DefineSqlTable_fromFile)

  fn = out['fn']
  sqlTableName = out['sqlTableName']
  sqlStr = out['sqlStr']

  # stage 1
  seqDf = CreateSeqTab_addEventCombin(
    df=df,
    seqDimCols=seqDimCols,
    partitionCols0=partitionCols0,
    sliceCols=sliceCols,
    seqPropCols=seqPropCols,
    timeGap=timeGap,
    timeCol=timeCol,
    trim=trim,
    countDistinctCols=countDistinctCols,
    seqCol=seqCol,
    noShifted=noShifted,
    timeColEnd=timeColEnd,
    addSeqPropMixedCols=addSeqPropMixedCols,
    ordered=ordered,
    fn=fn,
    writePath=writePath)

  print(seqDf[:2])

  '''
  seqCountDf = CalcSeqCountWrtId(
    seqDf=seqDf,
    trim=trim,
    sliceCols=sliceCols,
    seqPropCols=map(lambda x: x + '_parallel', ['form_factor']),
    seqCol='seq',
    countDistinctCols=countDistinctCols,
    seqLenCol=None)
  '''

  if writeTableLogFn != None:
    df = pd.DataFrame({
      'dataNames':[';'.join(dataNamesList)],
      'dataDesc':[dataDesc],
      'seqDimCols': [';'.join(seqDimCols)],
      'countDistinctCols': [';'.join(countDistinctCols)],
      'seqCol': [seqCol],
      'noShifted': [noShifted],
      'condDict':[DictOfLists_toString(condDict, noneString='None')],
      'trim':[trim],
      'seqPropCols':[';'.join(seqPropCols)],
      'sliceCols':[';'.join(sliceCols)],
      'writePath':[writePath],
      'fileName':[fn],
      'sqlTableName':[sqlTableName],
      'sqlStr':[sqlStr]
      })

    cols = [
        'dataNames', 'dataDesc', 'seqDimCols', 'countDistinctCols',
        'seqCol', 'noShifted', 'condDict', 'trim', 'seqPropCols',
        'sliceCols', 'writePath', 'fileName', 'sqlTableName', 'sqlStr']

    df = df[cols].copy()

    if FileExists(writeTableLogFn):
      df0 = ReadCsv(fn=writeTableLogFn)
      df = df.append(df0)

    df = df[cols].copy()

    Mark(
        df,
        text='updated built sql tables names for finding sig sequences',
        color='purple',
        bold=True)

    df = df.drop_duplicates()
    Mark(list(df.columns), 'after append')
    df = df.sort_values(['fileName'])

    WriteCsv(fn=writeTableLogFn, df=df, sep=',', append=False)

  if defineSqlTab:
    Mark(sqlStr, '\n this sql table was defined/updated on the fly \n')
    ExecSqlQuery(query=sqlStr)
  return {
      'fn':fn, 'sqlTableName':sqlTableName, 'sqlStr':sqlStr, 'seqDf':seqDf}

## write sig seq results to appropriate place
def Write_sigSeqDf(
    sigDf, sqlTableName, path, regDict, condDict):

  for col in ['level_0',  'index']:
    if col in sigDf.columns:
      del sigDf[col]
  fn0 = sqlTableName.replace('.', '_') + '_significant_seq'
  if len(regDict) > 0:
    fn0 = fn0 + '_regex_' + DictOfLists_toString(regDict)
  if len(condDict) > 0:
    fn0 = fn0 + '_cond_' + DictOfLists_toString(condDict)
  fn = path + fn0 + '.csv'

  sigDf['seq'] = sigDf['seq'].map(lambda x: x.replace('>', ' > '))
  sigDf = sigDf.sort_values(by=['seq', 'slice_comb'])

  WriteCsv(fn=fn, df=sigDf)
