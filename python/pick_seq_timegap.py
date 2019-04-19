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

""" This is code to find an appropriate timegap length
for defining sequences (eg user journeys)
by inspecting the distribution of timegaps between events
with the same "products" or varying products
"""

## Getting the distribution of timeGaps between
# events of same type and differing ones (eg watchFeat -> watchFeat vs watchFeat -> search)
# df is the dataframe
# respCol is the property/event we like to track
# timeCol is the column with the time stamp
# unitCol is the partitioning column: e.g. user_id

def TimeGapDistbn(
   df, respCol, timeCol, unitCol, minGap, minSliceSs, sep=" > "):

  df['delta'] = (df[timeCol] - df[timeCol].shift()).fillna(0)
  df['delta_sec'] = df['delta'].values / np.timedelta64(1, 's')
  df['pair'] = (df[respCol] + '---' + df[respCol].shift()).fillna('')
  df['pair'] = df['pair'].str.split('---')
  df['pair'] = df['pair'].map(lambda x: list(set(x)))
  df[unitCol + '_switch'] = (
      df[unitCol] + '---' + df[unitCol].shift()).fillna('')
  df[unitCol + '_switch'] =df[unitCol + '_switch'].str.split('---')
  df[unitCol + '_switch'] = df[unitCol + '_switch'].map(lambda x: list(set(x)))
  df = df[df[unitCol + '_switch'].map(len) < 2]
  df = df[df['delta_sec'] < minGap]
  indSame = df['pair'].map(len) < 2
  indSwitch = df['pair'].map(len) >= 2
  df['usage'] = df['pair'].map(lambda x: sep.join(x))

  def PlotandAgg(df0, pltTitle=''):

    df0['delta_sec'] = df0['delta_sec'] + 0.2
    g = df0.groupby(['usage'])['usage']
    tab = g.agg(len)
    values = list(tab[tab > minSliceSs].keys())
    df1 = df0[df0['usage'].isin(values)]
    df1.boxplot(column=['delta_sec'], by=['usage'])
    plt.yscale('log', nonposy='clip')
    plt.xticks(rotation='vertical')
    axes = plt.gca()
    axes.set_ylim([0, None])
    fig = plt.gcf()
    #fig.set_size_inches(15, 10)
    plt.axhline(y=100, hold=None, alpha=0.2)
    plt.axhline(y=0, hold=None, alpha=0.2)
    plt.title(pltTitle)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    g = df1.groupby(['usage'], as_index=False)
    dfAgg = g.agg({
      'delta_sec': {
          'Mean': np.mean,
          'Q5': QuantileFcn(5),
          'Q25': QuantileFcn(25),
          'Q50': QuantileFcn(50),
          'Q75': QuantileFcn(75),
          'Q95': QuantileFcn(95)
      }
    })

    dfAgg.columns = [''.join(col).strip() for col in dfAgg.columns.values]
    cols = (
        ['usage']
        + list('delta_sec' + pd.Series(['Q5', 'Q25', 'Q50', 'Q75', 'Q95'])))

    out = dfAgg[cols]
    return {'df': df1, 'dfAgg': dfAgg}

  plt.figure()
  same = PlotandAgg(df0=df[indSame], pltTitle='same')
  plt.figure()
  switch = PlotandAgg(df0=df[indSwitch], pltTitle='switch')

  outDict = {
      'sameDf': same['df'],
      'sameDfAgg': same['dfAgg'],
      'switchDf': switch['df'],
      'switchDfAgg': switch['dfAgg']
  }

  return outDict

'''
df = Sim_depUsageData(userNum=5, subSeqLen=3, repeatPattern=200)
'''
