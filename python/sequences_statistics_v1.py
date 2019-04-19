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

################# Calculate Statistics on sequences ######################

## this generates a function which calculates a value
# (e.g. a probability or odds)
# for each given input (usually a seq s) using a seq data frame
# this function uses the count column to exaggerate
def SeqDomainedFcn(
    df,
    seqCol,
    fcnDict,
    Compose=None,
    seqCountCol=None,
    SubsetDfFcn=None,
    sepStr=None,
    seqLengthMin=1):

  if Compose == None:
    def Compose(u):
      return(u)
  df2 = df.copy()
  if sepStr != None:
    df2 = df2.assign(**{seqCol: df2[seqCol].str.split(sepStr)})
  if seqCountCol == None:
    seqCountCol = 'seq_count'
    df2[seqCountCol] = 1
  ## we drop sequences whose length are less than 2
  df2 = df2[df2[seqCol].apply(len) >= seqLengthMin]

  def Out(x):
    df3 = df2.copy()
    if SubsetDfFcn != None:
      df3 = SubsetDfFcn(x)(df3)
    valueDict = {}
    fcnNames = fcnDict.keys()
    for key in fcnNames:
      F = fcnDict[key]
      vec = (np.array([F(x)(u) for u in df3[seqCol].values]) *
        np.array(df3[seqCountCol].values))
      valueDict[key] = vec.sum()
    out = Compose(valueDict)
    return out

  return Out

## takes sequence data, turns it to sequences of length two
# (x1=Entry app, x2=Used app) only if needed
# constructs a function which calculates
# for some pairs we may want to restrict the calculation to a subsample
# for example we may want to make sure that the seq considered come from users
# who have indeed used both prods in the pair
def SeqTransOddsFcn(df,
                    seqCol,
                    seqCountCol=None,
                    SubsetDfFcn=None,
                    sepStr=None,
                    seqLengthMin=2):

  def BothFcn(x):
    x = x.split(sepStr)
    if len(x) < 2:
      warnings.warn('Warning: length of x was less than 3. BothFcn Err.')
      def F(l):
        return(False)
      return F
    def F(l):
      if len(l) < 2:
        return(False)
      return(x[0:2] == l[0:2])
    return F

  def PreFcn(x):
    x = x.split(sepStr)
    def F(l):
      return x[0] == l[0]
    return F

  def PostFcn(x):
    x = x.split(sepStr)
    if len(x) < 1:
      warnings.warn('Warning: length of x was less than 3. PostFcn Err.')
      def F(l):
        return(False)

    def F(l):
      if len(l) < 2:
        return(False)
      return x[1] == l[1]
    return F

  def SsFcn(x):
    def F(l):
      return 1
    return F

  fcnDict = {}
  fcnDict['preNum'] = PreFcn
  fcnDict['postNum'] = PostFcn
  fcnDict['bothNum'] = BothFcn
  fcnDict['ss'] = SsFcn

  def Compose(d):
    p = np.nan
    if (d['preNum'] * d['postNum'] > 0):
      p = 1.0 * d['ss'] * d['bothNum'] / (d['preNum'] * d['postNum'])
    return (p)

  Out = SeqDomainedFcn(
      df=df,
      seqCol=seqCol,
      fcnDict=fcnDict,
      Compose=Compose,
      seqCountCol=seqCountCol,
      SubsetDfFcn=SubsetDfFcn,
      sepStr=sepStr,
      seqLengthMin=seqLengthMin)
  return(Out)

'''
df = GenUsageData(userNum=50, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
  dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
ElemsExists = AddMembershipColFcn(setDf=setDf, setCol='prod')
SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod',
partitionCols=['user_id'])
def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))
#Mark(SubsetDfFcn('>'.join(pair))(df=df)['user_id'].value_counts())
seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
  timeCol='time', timeGap=2*60, trim=5, keepIndCols=True, ordered=False)
seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=3, sepStr='>')
GetOdds = SeqTransOddsFcn(df=seqDf.copy(), seqCol='seq',
seqCountCol='seq_count',
  SubsetDfFcn=None, sepStr='>')
GetOdds_subset = SeqTransOddsFcn(df=seqDf.copy(), seqCol='seq',
  seqCountCol='seq_count', SubsetDfFcn=SubsetDfFcn, sepStr='>')
#print(seqDf[['seq', 'seq_count']])

def H(s):
  Mark(s)
  print(ElemsExists(s.split('>'))['elems_exist'].mean())
  Mark([GetOdds(s), GetOdds_subset(s)])
  Mark([GetOdds2(s), GetOdds_subset2(s)])

H('editingFeat>photoFeat')
H('browsingFeat>photoFeat')
H('editingFeat>watchFeat')
H('watchFeat>editingFeat')
H('photoFeat>watchFeat')
H('watchFeat>photoFeat')
H('watchFeat>watchFeat')
'''

## calculates a "triple probability"
## this is defined to be P(X1=a, X2=b, X3=c) / P(X1=a)*P(X2=b)*P(X1=c)
def SeqTripleOddsFcn(df,
                     seqCol,
                     seqCountCol=None,
                     SubsetDfFcn=None,
                     sepStr=None,
                     seqLengthMin=3):

  def TripleMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 3:
      warnings.warn('Warning: len(x) less than 3, in SeqTripleOddsFcn')
      def F(l):
        return False
      return F

    def F(l):
      if len(l) < 3:
        return False
      return x[0:3] == l[0:3]
    return F

  def FirstMatchFcn(x):
    x = x.split(sepStr)
    def F(l):
      return x[0] == l[0]
    return F

  def SecMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 2:
      warnings.warn('Warning: length of x was less than 3 in SecMatchFcn.')
      def F(l):
        return False
      return F
    def F(l):
      if len(l) < 2:
        return False
      return x[1] == l[1]
    return F

  def ThirdMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 3:
      warnings.warn('Warning: length of x was less than 3 in ThirdMatchFcn.')
      def F(l):
        return False
      return F
    def F(l):
      if len(l) < 3:
        return False
      return x[2] == l[2]
    return F

  def SsFcn(x):
    def F(l):
      return 1
    return F

  fcnDict = {}
  fcnDict['tripleMatch'] = TripleMatchFcn
  fcnDict['firstMatch'] = FirstMatchFcn
  fcnDict['secondMatch'] = SecMatchFcn
  fcnDict['thirdMatch'] = ThirdMatchFcn
  fcnDict['ss'] = SsFcn

  def Compose(d):
    p = np.nan
    denom = 1.0 * d['firstMatch'] * d['secondMatch'] * d['thirdMatch']
    if denom > 0:
      p = 1.0 * (d['ss']**2) * d['tripleMatch'] / denom
    return p

  Out = SeqDomainedFcn(
      df=df,
      seqCol=seqCol,
      fcnDict=fcnDict,
      Compose=Compose,
      seqCountCol=seqCountCol,
      SubsetDfFcn=SubsetDfFcn,
      sepStr=sepStr,
      seqLengthMin=seqLengthMin)

  return Out

'''
df = GenUsageData(userNum=100, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
  dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
ElemsExists = AddMembershipColFcn(setDf=setDf, setCol='prod')
SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod',
partitionCols=['user_id'])
def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))
#Mark(SubsetDfFcn('>'.join(pair))(df=df)['user_id'].value_counts())
seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
  timeCol='time', timeGap=2*60, trim=5, keepIndCols=True, ordered=False)

seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=3, sepStr='>')
GetTripleOdds = SeqTripleOddsFcn(df=seqDf.copy(), seqCol='seq',
seqCountCol='seq_count',
  SubsetDfFcn=None, sepStr='>')
GetTripleOdds_subset = SeqTripleOddsFcn(df=seqDf.copy(), seqCol='seq',
  seqCountCol='seq_count', SubsetDfFcn=SubsetDfFcn, sepStr='>')
#print(seqDf[['seq', 'seq_count']])

def H(s):
  Mark(s)
  print(ElemsExists(s.split('>'))['elems_exist'].mean())
  Mark([GetTripleOdds(s), GetTripleOdds_subset(s)])

H('editingFeat>photoFeat>editingFeat')
H('browsingFeat>photoFeat>browsingFeat')
H('editingFeat>watchFeat>editingFeat')
H('watchFeat>editingFeat>watchFeat')
'''

def SeqQuadOddsFcn(
    df,
    seqCol,
    seqCountCol=None,
    SubsetDfFcn=None,
    sepStr=None,
    seqLengthMin=4):


  def QuadMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 4:
      warnings.warn("Warning: len(x) less that 4, SeqQuadOddsFcn err.")
      def F(l):
        return False
      return F

    def F(l):
      if len(l) < 4:
        return(False)
      return x[0:4] == l[0:4]
    return F

  def FirstMatchFcn(x):
    x = x.split(sepStr)
    def F(l):
      return (x[0] == l[0])
    return F

  def SecMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 2:
      warnings.warn('Warning: len(x) less than 3. SecMatchFcn err.')
      def F(l):
        return False
      return F
    def F(l):
      if len(l) < 2:
        return False
      return x[1] == l[1]
    return F

  def ThirdMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 3:
      warnings.warn('Warning: len(x) less than 3. ThirdMatchFcn err.')
      def F(l):
        return False
      return F
    def F(l):
      if len(l) < 3:
        return False
      return x[2] == l[2]
    return F

  def FourthMatchFcn(x):
    x = x.split(sepStr)
    if len(x) < 4:
      warnings.warn('Warning: len(x) less than 4. FourthMatchFcn err.')
      def F(l):
        return False
      return F
    def F(l):
      if len(l) < 4:
        return False
      return x[3] == l[3]
    return F

  def SsFcn(x):
    def F(l):
      return(1)
    return(F)

  fcnDict = {}
  fcnDict['quadMatch'] = QuadMatchFcn
  fcnDict['firstMatch'] = FirstMatchFcn
  fcnDict['secondMatch'] = SecMatchFcn
  fcnDict['thirdMatch'] = ThirdMatchFcn
  fcnDict['fourthMatch'] = FourthMatchFcn
  fcnDict['ss'] = SsFcn

  def Compose(d):
    p = np.nan
    denom = (1.0 * d['firstMatch'] * d['secondMatch'] * d['thirdMatch'] *
             d['fourthMatch'])
    if denom > 0:
      p = 1.0 * (d['ss']**3) * d['quadMatch'] / (denom * 1.0)
    return p

  Out = SeqDomainedFcn(
      df=df,
      seqCol=seqCol,
      fcnDict=fcnDict,
      Compose=Compose,
      seqCountCol=seqCountCol,
      SubsetDfFcn=SubsetDfFcn,
      sepStr=sepStr,
      seqLengthMin=seqLengthMin)

  return Out

'''
df = GenUsageData(userNum=200, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
             dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
ElemsExists = AddMembershipColFcn(setDf=setDf, setCol='prod')
SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod', partitionCols=['user_id'])
def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))
#Mark(SubsetDfFcn('>'.join(pair))(df=df)['user_id'].value_counts())
seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
  timeCol='time', timeGap=3*60, trim=4, keepIndCols=True, ordered=False)

seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=4, sepStr='>')

GetOdds = SeqQuadOddsFcn(df=seqDf.copy(), seqCol='seq',
                         seqCountCol='seq_count',
                         SubsetDfFcn=None, sepStr='>')
GetOdds_subset = SeqQuadOddsFcn(df=seqDf.copy(), seqCol='seq',
  seqCountCol='seq_count', SubsetDfFcn=SubsetDfFcn, sepStr='>')
#print(seqDf[['seq', 'seq_count']])

def H(s):
  Mark(s)
  print(ElemsExists(s.split('>'))['elems_exist'].mean())
  Mark([GetOdds(s), GetOdds_subset(s)])

H('editingFeat>photoFeat>editingFeat>photoFeat')
H('browsingFeat>photoFeat>browsingFeat>photoFeat')
H('editingFeat>watchFeat>editingFeat>watchFeat')
H('watchFeat>editingFeat>watchFeat>editingFeat')
H('editingFeat>mailingFeat>locFeat>browsingFeat_IMAGES')
H('watchFeat_MUSIC>PresFeat>watchFeat_MUSIC>SIGN_IN')
H('watchFeat>browsingFeat_IMAGES>PresFeat>watchFeat_MUSIC')
H('PresFeat>browsingFeat_IMAGES>PresFeat>exploreFeat')

seqDf2 = seqDf[seqDf['seq'].apply(lambda x: x.count('>')) == 3]
obsSeq = list(set(seqDf['seq'].values))
obsSeq

H(obsSeq[0])
H(obsSeq[1])
'''

## define one function to capture all three functions
def SeqRelativeProbFcn(seqLengthLimit):

  outDict = {
    1: None,
    2: SeqTransOddsFcn,
    3: SeqTripleOddsFcn,
    4: SeqQuadOddsFcn}

  return outDict[seqLengthLimit]

## this function works on data frame and calculates confidence intervals
# for all appearing transitions
# (not implemented for all possible transitions (cartesian prod)
# yet since that does not seem to be useful)
# its for transitions which appear at least once in the data
# SubsetDf is a function that subsets the data frame for each give pair:
# Subset(df, pre, post)
def SeqConfIntDf(df,
                 seqCol,
                 Fcn,
                 seqCountCol=None,
                 SubsetDfFcn=None,
                 shift=True,
                 trim=2,
                 sepStr=None,
                 bsSize=None,
                 valueColName='value',
                 seqLengthMin=1,
                 seqCountMin=10):

  df2 = df.copy()
  if shift:
    df2 = ShiftedSeqDf(df=df2, seqCol=seqCol, k=trim, sepStr=sepStr)
  ## dropping sequences whose length are less than 2
  df2 = df2[df2[seqCol].apply(lambda x: x.count(sepStr)) >= (seqLengthMin - 1)]
  outDf = df2[[seqCol, seqCountCol]]
  outDf = outDf.drop_duplicates()

  seqSet = outDf[seqCol].values

  def ValuesDf(df):

    G = Fcn(
        df=df,
        seqCol=seqCol,
        seqCountCol=seqCountCol,
        SubsetDfFcn=SubsetDfFcn,
        sepStr=sepStr,
        seqLengthMin=seqLengthMin)
    values = [G(x) for x in seqSet]

    return values

  values = ValuesDf(df2)
  outDf[valueColName] = values

  if bsSize != None:
    outDf2 = outDf.copy()
    n = df.shape[0]
    def Bs():
      df0 = BsWithCounts(df=df, countCol=seqCountCol)
      odds = ValuesDf(df0)
      return(odds)
    colList = []
    for i in range(bsSize):
      col = 'odds' + str(i)
      outDf2[col] = Bs()
      colList.append(col)
    ## we replace np.nan with 1 to reflect the true uncertainty and fix coverage
    # this is deprecated
    #outDf2 = outDf2.astype(object).replace(np.nan, 1)
    def Lower(row):
      return row[colList].quantile(0.05)
    def Upper(row):
      return row[colList].quantile(0.95)
    outDf[valueColName + '_CI_lower'] = outDf2.apply(Lower, axis=1)
    outDf[valueColName + '_CI_upper'] = outDf2.apply(Upper, axis=1)

    ## reset CI values if the seq is observed only 1-4 times
    outDf[valueColName + '_CI_lower'][outDf[seqCountCol] < seqCountMin] = 0
    outDf[valueColName + '_CI_upper'][outDf[seqCountCol] < seqCountMin] =  (
        outDf[valueColName][outDf[seqCountCol] < seqCountMin])*2 + 10

  return outDf


'''
size = 10
x = np.random.poisson(lam=1.0, size=size)
x = 1
df = pd.DataFrame({
    'seq':np.random.choice(a=['a>k>b>k>d>b>k>ba>b>a>k>a>b>f>h>k',
                                   'b>c>f>k>f>h>b>a>d>a>l>b>a>b',
                                   'f>k>v>f>g>a>b>d>a>b>a>b',
                                   'f>h>k>b>a>e>k>a>b'],
                                size=size,
                                replace=True),
    'seq_count':x,
    'col2':np.random.uniform(low=0.0, high=100.0, size=size).round(0),
    'col3':np.random.uniform(low=0.0, high=100.0, size=size).round(0),
    'col4':np.random.uniform(low=0.0, high=100.0, size=size).round(0)})

seqDf = ShiftedSeqDf(df=df, seqCol='seq', k=3, sepStr='>')
#seqDf2 = AddSeqOrdEvent(df=seqDf, seqCol='seq', sepStr='>')
#print(seqDf)
Fcn = SeqTransOddsFcn(df=seqDf, seqCol='seq',
                      seqCountCol='seq_count', sepStr='>')
print(Fcn('a>b'))
print(Fcn('b>x'))
SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=SeqTransOddsFcn, seqCountCol=None,
  shift=True, sepStr='>', bsSize=5, seqLengthMin=2)

## Example 2
df = GenUsageData(userNum=8, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
             dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))
setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
AddMembershipCol = AddMembershipColFcn(setDf=setDf, setCol='prod')
SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod',
                                     partitionCols=['user_id'])
def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))

pair = [browsingFeat, 'watchFeat']
Mark(AddMembershipCol(pair))
Mark(SubsetDfFcn0(subSet=pair)(df)['user_id'].value_counts())

seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
  timeCol='time', timeGap=2*60, trim=2, keepIndCols=True, ordered=False)
seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=3, sepStr='>')
out1 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=SeqTransOddsFcn,
  seqCountCol=None, SubsetDfFcn=None, shift=True, sepStr='>', bsSize=5,
                    seqLengthMin=2)
out2 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=SeqTransOddsFcn,
  seqCountCol=None, SubsetDfFcn=SubsetDfFcn, shift=True, sepStr='>', bsSize=5,
                    seqLengthMin=2)
#print(out2)

## Example 3:

df = GenUsageData(userNum=10, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
             dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
AddMembershipCol = AddMembershipColFcn(setDf=setDf, setCol='prod')

SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod',
                                     partitionCols=['user_id'])
def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))

pair = [browsingFeat, 'watchFeat']
Mark(AddMembershipCol(pair))
Mark(SubsetDfFcn0(pair)(df)['user_id'].value_counts())

seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
                               timeCol='time', timeGap=2*60, trim=2,
                               keepIndCols=True, ordered=False)
seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=3, sepStr='>')

F = SeqTransOddsFcn(df=seqDf.copy(), seqCol='seq',
                    seqCountCol='seq_count', SubsetDfFcn=None, sepStr='>')
G = SeqTransOddsFcn(df=seqDf.copy(), seqCol='seq',
                    seqCountCol='seq_count', SubsetDfFcn=SubsetDfFcn,
                    sepStr='>')

print(seqDf[['seq', 'seq_count']])
def H(pair):
  print([F(pair[0] + '>' + pair[1]), G(pair[0] + '>' + pair[1])])

H(['photoFeat', browsingFeat])
H([browsingFeat, 'photoFeat'])
H(['editingFeat', 'watchFeat'])
bsSize = 10

out1 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=SeqTransOddsFcn,
                    seqCountCol=None, SubsetDfFcn=None, shift=True, trim=2,
                    sepStr='>', bsSize=bsSize, valueColName='value',
                    seqLengthMin=2)

out2 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=SeqTransOddsFcn,
                    seqCountCol=None, SubsetDfFcn=SubsetDfFcn, shift=True,
                    trim=2, sepStr='>', bsSize=bsSize, valueColName='value',
                    seqLengthMin=2)


compareDf = pd.merge(out1, out2, on=['seq'], how='outer')
plt.scatter(compareDf['value_x'], compareDf['value_y'])
'''


## calculate the followup odds (relative prob)
def SeqSigValueDf(df,
                  timeCol,
                  timeGap,
                  seqDimCols,
                  Fcn,
                  trim,
                  partitionCols=[],
                  sliceCols=[],
                  checkElemsExist=False,
                  condDict=None,
                  lowerThresh=1.25,
                  upperThresh=0.8,
                  valueColName='value',
                  seqLengthMin=1,
                  ordered=False,
                  TransResDfList=[],
                  seqCountMin=10,
                  fn0=None,
                  writePath=''):

  def StrReplace(s):
    return(s.replace('>', ' > '))

  seqDf = CreateSeqDf(
      df=df,
      timeCol=timeCol,
      seqDimCols=seqDimCols,
      partitionCols=partitionCols,
      timeGap=timeGap,
      trim=trim,
      ordered=ordered)

  # integrate seq_start_time and seq_length out since we don't need it
  seqTabDf = IntegOutDf(
    df=seqDf,
    integFcn=sum,
    integOutCols=['seq_shift_order', 'seq_length'],
    valueCols=['seq_count'])

  ## we make sure all the sequences if the data have the same length
  # by adding blanks
  # this will avoid unexpected dropping later on when calculating probabilities
  eventCols = ['event_1', 'event_2', 'event_3', 'event_4'][:seqLengthMin]
  seqTabDf['seq'] = seqTabDf['event_1']
  for i in range(1, seqLengthMin):
    seqTabDf['seq'] = seqTabDf['seq'] + '>' + seqTabDf[eventCols[i]]

  ## we check if the pair exists for the partitionCols
  SubsetDfFcn = None

  if checkElemsExist:
    setDf = GetSetIndCols(
        df=df, respCol='-'.join(seqDimCols), partitionCols=partitionCols)
    SubsetDfFcn0 = ElemsExist_subsetDfFcn(
        setDf=setDf,
       setCol='-'.join(seqDimCols),
       partitionCols=partitionCols)
    def SubsetDfFcn(x):
      l = x.split('>')
      return(SubsetDfFcn0(l))

  if len(sliceCols) == 0:
    seqTabDf['tempSliceCol'] = 0
    sliceCols = ['tempSliceCol']

  def CalcPerSlice(group):
    valueDf = SeqConfIntDf(
        df=group,
        seqCol='seq',
        Fcn=Fcn,
        seqCountCol='seq_count',
        SubsetDfFcn=SubsetDfFcn,
        shift=False,
        trim=trim,
        sepStr='>',
        bsSize=100,
        valueColName=valueColName,
        seqLengthMin=seqLengthMin,
        seqCountMin=seqCountMin)

    return valueDf

  g = seqTabDf.groupby(sliceCols)

  valueDf = g.apply(CalcPerSlice)
  valueDf = valueDf.reset_index()
  for i in range(len(TransResDfList)):
    F = TransResDfList[i]
    valueDf = F(valueDf)

  if 'tempSliceCol' in valueDf.columns:
    del valueDf['tempSliceCol']

  incDf = valueDf[valueDf[valueColName + '_CI_lower'] > lowerThresh]
  incDf = incDf.sort_values([valueColName + '_CI_lower'], ascending=False)
  incDf = incDf.round(1)
  decDf = valueDf[valueDf[valueColName + '_CI_upper'] < upperThresh]
  decDf = decDf.sort_values([valueColName + '_CI_upper'], ascending=True)
  decDf = decDf.round(2)
  sameDf = valueDf[(valueDf[valueColName + '_CI_upper'] > 1) *
                   (valueDf[valueColName + '_CI_lower'] < 1) *
                   (valueDf[valueColName + '_CI_upper'] -
                    valueDf[valueColName + '_CI_lower'] < 2)]
  sameDf['ci_width'] = (sameDf[valueColName + '_CI_upper'] -
                        sameDf[valueColName + '_CI_lower'])
  sameDf = sameDf.sort_values(['ci_width'])

  if fn0 != None:
    #incDf['seq'] = incDf['seq'].map(StrReplace)
    #decDf['seq'] = decDf['seq'].map(StrReplace)
    #sameDf['seq'] = sameDf['seq'].map(StrReplace)
    fn1 = fn0 + '_inc.csv'
    fn2 = fn0 + '_dec.csv'
    #fn3 = fn0 + '_same.csv'
    WriteCsv(fn=writePath+fn1, df=incDf, printLog=True)
    WriteCsv(fn=writePath+fn2, df=decDf, printLog=True)
    #WriteCsv(fn=writePath+fn3, df=sameDf)
  outDict = {
      'seqTabDf': seqTabDf,
      'incDf': incDf,
      'decDf': decDf,
      'sameDf': sameDf,
      'seqTabDf': seqTabDf,
      'valueDf': valueDf
  }

  return outDict

'''

df = GenUsageData(userNum=10, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
             dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
AddMembershipCol = AddMembershipColFcn(setDf=setDf, setCol='prod')
SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod', partitionCols=['user_id'])
def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))
pair = [browsingFeat, 'watchFeat']
Mark(AddMembershipCol(pair))
Mark(SubsetDfFcn0(pair)(df)['user_id'].value_counts())
seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
                               timeCol='time', timeGap=2*60, trim=2,
                               keepIndCols=True, ordered=False)
seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=3, sepStr='>')
F = SeqTransOddsFcn(df=seqDf.copy(), seqCol='seq',
                    seqCountCol='seq_count', SubsetDfFcn=None, sepStr='>')
G = SeqTransOddsFcn(df=seqDf.copy(), seqCol='seq',
                    seqCountCol='seq_count', SubsetDfFcn=SubsetDfFcn,
                    sepStr='>')
#Mark(seqDf[['seq', 'seq_count']])
def H(pair):
  print([F(pair[0] + '>' + pair[1]), G(pair[0] + '>' + pair[1])])

H(['photoFeat', browsingFeat])
H([browsingFeat, 'photoFeat'])
H(['editingFeat', 'watchFeat'])
bsSize = 10
Fcn = SeqTransOddsFcn

out1 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=Fcn, seqCountCol=None,
  SubsetDfFcn=None, shift=True, trim=2, sepStr='>', bsSize=bsSize,
                    valueColName='value', seqLengthMin=2)
out2 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=Fcn, seqCountCol=None,
                    SubsetDfFcn=SubsetDfFcn, shift=True, trim=2, sepStr='>',
                    bsSize=bsSize, valueColName='value', seqLengthMin=2)
compareDf = pd.merge(out1, out2, on=['seq'], how='outer')
plt.scatter(compareDf['value_x'], compareDf['value_y'])


sigDf = SeqSigValueDf(df=df, timeCol='time', timeGap=2*60, seqDimCols=['prod'],
                      Fcn=Fcn, trim=2, partitionCols=['user_id'], keepIndCols=False,
                      initBlankValue=None, lastBlankValue=None,
                      checkElemsExist=False, condDict=None, lowerThresh=1.25,
                      upperThresh=0.8, valueColName='value', TransDfList=None,
                      seqLengthMin=2, fn0=None)

'''


'''

df = GenUsageData(userNum=200, dt1=datetime.datetime(2017, 4, 12, 0, 0, 0),
             dt2=datetime.datetime(2017, 4, 12, 1, 0, 0))

setDf = GetSetIndCols(df=df, respCol='prod', partitionCols=['user_id'])
#print(setDf)
AddMembershipCol = AddMembershipColFcn(setDf=setDf, setCol='prod')

SubsetDfFcn0 = ElemsExist_subsetDfFcn(setDf=setDf, setCol='prod',
                                     partitionCols=['user_id'])

def SubsetDfFcn(x):
  l = x.split('>')
  return(SubsetDfFcn0(l))

basket = [browsingFeat, 'watchFeat', 'exploreFeat', 'editingFeat']
Mark(AddMembershipCol(basket))
Mark(SubsetDfFcn0(basket)(df)['user_id'].value_counts())

seqDf = CreateSeqTableContains(df=df, partitionCols=['user_id'], respCol='prod',
                               timeCol='time', timeGap=2*60, trim=None,
                               keepIndCols=True, ordered=False)

seqDf = ShiftedSeqDf(df=seqDf, seqCol='seq', k=4, sepStr='>')

F = SeqQuadOddsFcn(df=seqDf.copy(), seqCol='seq',
                   seqCountCol='seq_count', SubsetDfFcn=None, sepStr='>')
G = SeqQuadOddsFcn(df=seqDf.copy(), seqCol='seq',
                   seqCountCol='seq_count', SubsetDfFcn=SubsetDfFcn,
                   sepStr='>')
#Mark(seqDf[['seq', 'seq_count']])
def H(x):
  print(x)
  print([F(x), G(x)])

H('editingFeat>photoFeat>editingFeat>photoFeat')
H('browsingFeat>photoFeat>browsingFeat>photoFeat')
H('editingFeat>watchFeat>editingFeat>watchFeat')
H('watchFeat>editingFeat>watchFeat>editingFeat')
H('editingFeat>mailingFeat>locFeat>browsingFeat_IMAGES')
H('watchFeat_MUSIC>PresFeat>watchFeat_MUSIC>SIGN_IN')
H('watchFeat>browsingFeat_IMAGES>PresFeat>watchFeat_MUSIC')
H('PresFeat>browsingFeat_IMAGES>PresFeat>exploreFeat')

seqDf2 = seqDf[seqDf['seq'].apply(lambda x: x.count('>')) == 3]
obsSeq = list(set(seqDf['seq'].values))
obsSeq

H(obsSeq[0])
H(obsSeq[1])


bsSize = 10
Fcn = SeqQuadOddsFcn
out1 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=Fcn, seqCountCol=None,
                    SubsetDfFcn=None, shift=True, trim=4, sepStr='>',
                    bsSize=bsSize, valueColName='value', seqLengthMin=4)

out2 = SeqConfIntDf(df=seqDf, seqCol='seq', Fcn=Fcn, seqCountCol=None,
                    SubsetDfFcn=SubsetDfFcn, shift=True, trim=4, sepStr='>',
                    bsSize=bsSize, valueColName='value', seqLengthMin=4)
compareDf = pd.merge(out1, out2, on=['seq'], how='outer')
plt.scatter(compareDf['value_x'], compareDf['value_y'])

sigDf = SeqSigValueDf(df=df, timeCol='time', timeGap=2*60, seqDimCols=['prod'],
                      Fcn=Fcn, trim=4, partitionCols=['user_id'],
                      keepIndCols=False, initBlankValue=None,
                      lastBlankValue=None, checkElemsExist=False, condDict=None,
                      lowerThresh=1.25, upperThresh=0.8, valueColName='value',
                      TransDfList=None, seqLengthMin=4, fn0=None)


'''
