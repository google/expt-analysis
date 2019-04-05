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



""" These are functions for comparing bootstrap and asymptotic
to calculate CIs
these functions are not needed to perform actual analysis."""

## build bootstrap and asympt CI
def TestSigSeqMethods(
    dfRaw,
    partitionCols,
    partitionColsAgg,
    timeGap,
    timeCol,
    seqDimCols,
    sliceCols,
    trim,
    seqCountMin=20):

  ## asympt
  tic = time.clock()
  out = CreateSeqTable_withCond(
      df=dfRaw,
      timeCol=timeCol,
      seqDimCols=seqDimCols,
      partitionCols=partitionCols,
      timeGap=timeGap,
      trim=trim,
      keepIndCols=True,
      partitionColsAgg=partitionColsAgg,
      condDict=None,
      initBlankValue=None,
      lastBlankValue=None,
      extraCols=[],
      ordered=True)

  seqTabDf = out['shifted']
  Mark(seqTabDf[:5], 'seqTabDf directly (used in asympt method)')

  eventCols = ['event_1', 'event_2', 'event_3', 'event_4'][:trim]
  seqTabDf['seq'] = seqTabDf['event_1']
  for i in range(1, trim):
    seqTabDf['seq'] = seqTabDf['seq'] + '>' + seqTabDf[eventCols[i]]

  seqDfWithTotals = AddSeqSummableMetrics(
      seqDf=seqTabDf,
      sliceCols=sliceCols,
      seqPropCols=[],
      seqCol='seq',
      seqCountCol='seq_count')

  Mark(seqDfWithTotals[:5], 'seqDfWithTotals')
  seqDfWithSignif = AddSeqProbCiMetrics(
      seqDf=seqDfWithTotals,
      trim=trim,
      seqCountMin=seqCountMin)

  toc = time.clock()
  Mark(toc - tic, 'asympt method time in sec')

  ## bootstrap method
  tic = time.clock()
  Fcn = SeqRelativeProbFcn(trim)
  sigDf = SeqSigValueDf(
      df=dfRaw,
      timeCol=timeCol,
      timeGap=timeGap,
      seqDimCols=seqDimCols,
      Fcn=Fcn,
      trim=trim,
      partitionCols=partitionCols,
      keepIndCols=True,
      sliceCols=sliceCols,
      initBlankValue=None,
      lastBlankValue=None,
      checkElemsExist=False,
      condDict=None,
      lowerThresh=1.1,
      upperThresh=0.9,
      valueColName='value',
      TransDfList=None,
      seqLengthMin=trim,
      seqCountMin=seqCountMin,
      fn0=None)
  toc = time.clock()
  Mark(toc - tic, 'bs method time in sec')
  seqTabDf = sigDf['seqTabDf']
  Mark(seqTabDf[:10], 'seqTabDf from bs method')


  return {'bs': sigDf, 'asympt':seqDfWithSignif, 'summable': seqDfWithTotals}

## compare the BS and asympt methods
def TestPlotCompareSigMethods(
    sigDict, sliceCols=[], removeBlankSeqs=True, figSize=[30, 10]):

  sigDf = sigDict['bs']
  seqDfWithSignif = sigDict['asympt']

  dfBs = sigDf['valueDf']
  dfAsym = seqDfWithSignif.sort_values(['count'], ascending=[0])
  dfAsym = dfAsym[
      ['seq'] +
      sliceCols +
      ['relative_prob', 'relative_prob_lower', 'relative_prob_upper']]
  dfCompare = pd.merge(dfBs, dfAsym, on=['seq'] + sliceCols, how='outer')

  dfCompare = dfCompare.sort_values(['seq'] + sliceCols)

  #Mark(dfCompare, 'dfCompare before removing nan')
  dfCompare = dfCompare[dfCompare['value'].map(str) != 'nan']
  #Mark(dfCompare, 'dfCompare after removing nan')

  Mark(dfCompare.sort_values(['relative_prob_lower'], ascending=[0])[:8])

  dfCompare = Concat_stringColsDf(dfCompare, cols=['seq'] + sliceCols,
                            colName='seq_slice', sepStr='-')

  ## subset the cases where at least one of the methods is sig
  dfCompare2 = (
      dfCompare[(dfCompare['value_CI_lower'] > 1.2)
      + (dfCompare['relative_prob_lower'] > 1.2)])

  ## subset the cases where at least one of the intervals not super wide
  '''
  dfCompare2 = dfCompare2[
      ((dfCompare2['value_CI_upper'] -
        dfCompare2['value_CI_lower']) < 1000) +
      ((dfCompare2['relative_prob_upper'] -
        dfCompare2['relative_prob_lower']) < 1000)]
  '''

  if removeBlankSeqs:
    dfCompare2 = dfCompare2[dfCompare2['seq'].map(lambda x: 'BLANK' not in x)]

  n = len(dfCompare2)

  fig, ax = plt.subplots()
  fig.set_size_inches(figSize[0], figSize[1])

  plt.scatter(range(n), dfCompare2['value'], color='grey', alpha=0.8)
  plt.bar(left=range(n), height=dfCompare2['value_CI_upper'].values,
          bottom=dfCompare2['value_CI_lower'].values, color='red',
          width=0.2, alpha=0.5)
  plt.scatter(map(lambda x: (float(x) + 0.25), range(n)),
              dfCompare2['relative_prob'], color='blue', alpha=0.8)
  plt.bar(left=map(lambda x: (float(x) + 0.25), range(n)),
          height=dfCompare2['relative_prob_upper'].values,
          bottom=dfCompare2['relative_prob_lower'].values, color='blue',
          width=0.25, alpha=0.5)

  labels = [item.get_text() for item in ax.get_xticklabels()]
  labels = list(dfCompare2['seq_slice'].values)
  ax.set_xticklabels(labels)
  locs, labels = plt.xticks(map(lambda x: (float(x) + 0.5), range(n)), labels)
  plt.setp(labels, rotation=85, fontweight='bold')
  plt.plot(range((n+2)), [1]*(n+2), color='orange', alpha=0.5)
  plt.plot(range((n+2)), [0]*(n+2), color='grey', alpha=0.5)
  plt.yscale('log')

'''
###### Comparing BS and asympt
## example 1: simulated data
dfRaw = GenUsageData(
    userNum=20,
    dt1=datetime.datetime(2017, 4, 12, 23, 30, 0),
    dt2=datetime.datetime(2017, 4, 13, 1, 0, 0))


dfRaw['date'] = dfRaw['time'].dt.date
dfRaw['prod'][:300] = ['PHOTOS', 'MAPS', 'GMAIL']*100

trim = 3


sigDict = TestSigSeqMethods(
    dfRaw=dfRaw,
    partitionCols=['user_id', 'date'],
    partitionColsAgg=['date'],
    timeGap=2*60,
    timeCol='time',
    seqDimCols=['prod'],
    sliceCols=[])

Mark(seqDfWithTotals.sort_values(['count'], ascending=[0])[:5])
Mark(seqDfWithSignif.sort_values(['count'], ascending=[0])[:5])
Mark(sigDf['incDf'][:5])
TestPlotCompareSigMethods(sigDict)

## example simulated with 4 seq
dfRaw = GenUsageData(
    userNum=30,
    dt1=datetime.datetime(2017, 4, 12, 23, 30, 0),
    dt2=datetime.datetime(2017, 4, 13, 3, 0, 0))


dfRaw['date'] = dfRaw['time'].dt.date
subSeqLen = ['DOCS', 'SHEETS', 'SLIDES', 'GMAIL']
k = 200
dfRaw['prod'][:(len(subSeqLen)*k)] = subSeqLen*k

trim = 4

sigDict = TestSigSeqMethods(
    dfRaw=dfRaw,
    partitionCols=['user_id', 'date'],
    partitionColsAgg=['date'],
    timeGap=2*60,
    timeCol='time',
    seqDimCols=['prod'],
    sliceCols=[],
    trim=trim)

Mark(seqDfWithTotals.sort_values(['count'], ascending=[0])[:5])
Mark(seqDfWithSignif.sort_values(['count'], ascending=[0])[:5])
Mark(sigDf['incDf'][:5])

TestPlotCompareSigMethods(sigDict, figSize=[15, 10])
'''
