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
This is code for simulation to compare metrics R1, R2 to discover sig sequences.
"""

## simulate the data, and inject subseq to data to test if methods can catch that
def SimulData(subSeq):

  dfRaw = GenUsageData(
      userNum=100,
      dt1=datetime.datetime(2017, 4, 11, 23, 0, 0),
      dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

  dfRaw['date'] = dfRaw['time'].dt.date

  Mark(dfRaw.shape)


  k = 50
  l = 6
  for i in range(k):
    x = subSeq + list(np.random.choice(
        a=['p1', 'p2', 'p3', 'p4', 'p5',
           'p6', 'p7'],
        size=(l-len(subSeq)),
        replace=True))

    dfRaw['prod'][l*i:l*(i+1)] = x

  trim = 3

  out = CreateSeqTable_withCond(
      df=dfRaw,
      timeCol='time',
      seqDimCols=['prod'],
      partitionCols=['user_id', 'date'],
      timeGap=10*60,
      trim=trim,
      keepIndCols=True,
      partitionColsAgg=['date'],
      condDict=None,
      initBlankValue=None,
      lastBlankValue=None,s
      extraCols=[],
      ordered=True)

  seqDf = out['shifted']

  sliceCols = ['date']

  seqDfWithTotals = AddSeqSummableMetrics(
      seqDf=seqDf,
      sliceCols=sliceCols,
      seqPropCols=[],
      seqCol='sequence',
      seqCountCol='sequence_count')

  seqDfWithSignif = AddSeqProbCiMetrics(
      seqDf=seqDfWithTotals,
      trim=trim)

  outDict = {'seqDfWithTotals': seqDfWithTotals,
             'seqDfWithSignif':seqDfWithSignif}

  return outDict



### plot the results
def DisplayResults(seqDfWithSignif):

  metrics = ['relative_prob_lower1', 'relative_prob_lower2']
  plt.scatter(seqDfWithSignif[metrics[0]], seqDfWithSignif[metrics[1]])
  #plt.xscale('log')
  #plt.yscale('log')
  m = max(seqDfWithSignif[metrics[0]].max(), seqDfWithSignif[metrics[1]].max())
  plt.plot([0, m], [0, m], 'k-', lw=2, alpha=0.4)
  plt.plot([0, m], [1, 1], 'k-', lw=2, alpha=0.3)
  plt.plot([1, 1], [0, m], 'k-', lw=2, alpha=0.3)
  plt.xlabel(metrics[0], fontsize=18)
  plt.ylabel(metrics[1], fontsize=16)

  df0 = seqDfWithSignif.sort_values(['relative_prob1'], ascending=[0])[[
      'sequence',
      #'event_1',
      #'event_2',
      #'event_3',
      #'event_1_count_agg',
      #'event_2_count_agg',
      #'event_3_count_agg',
      #'event_1_to_2_count_agg',
      #'event_2_to_3_count_agg',
      #'event_1_to_3_count_agg',
      'count',
      'relative_prob1',
      'relative_prob2',
      'relative_prob_lower1',
      'relative_prob_lower2']]

  Mark(df0[:10])



# simulation 1
subSeq = ['p1', 'p2', 'p3']
outDict = SimulData(subSeq=subSeq)
seqDfWithSignif = outDict['seqDfWithSignif']
DisplayResults(seqDfWithSignif)


# simulation 2
subSeq = ['p1', 'p2']
outDict = SimulData(subSeq=subSeq)
seqDfWithSignif = outDict['seqDfWithSignif']
DisplayResults(seqDfWithSignif)


sliceColsAgg = ['date']
df0 = Plt_sigSeq_compareSlices(
    seqDfWithSignif=seqDfWithSignif,
    sliceCols=sliceColsAgg,
    metricCol='relative_prob1',
    metricColLower='relative_prob_lower1',
    metricColUpper='relative_prob_upper1',
    condDict=None,
    removeBlankSeqs=True,
    relativeProbLowerLim = 1.01,
    orderByCol='relative_prob_lower1',
    seqNumLimit = None,
    rotation=90,
    figSize=[8, 10])
