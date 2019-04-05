
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

codePath = ""

source(paste0(codePath, "data_analysis.R"))
source(paste0(codePath, "clustering.R"))
source(paste0(codePath, "plotting.R"))

#@title simulate data and test the main function
varNum = 8 # param
sampleSize = 50 # param
# this is to generate a variance-covariance matrix via choleski decomp
l = matrix(0, varNum, varNum)
diag(l) = 1
for (i in 2:varNum) {
  l[i, ] = c(rnorm(i-1), 1, rep(0, varNum-i))
}
# this (sig) is the variance-covariance matrix of the underlying dist
# in case you like to compare
sig = t(l) %*% l
u = matrix(rnorm(sampleSize*varNum), sampleSize, varNum)
x = u %*% l
## this is to insure the response is always non-negative
x = x * (x >= 0)
df1 = data.frame(x)


clustNum = 10 # param
dfCl = OrderPlotClustCentMid(df=df1, centers=clustNum, method='kmeans')
PlotClustResults(dfCl)



#@title bonus: compare clustering with bucketing! how the clusters centers are correlated vs how the bucket centers are correlated?
source(src)
varNum = 10 # param
sampleSize = 500 # param
# this is to generate a variance-covariance matrix via choleski decomp
l = matrix(0, varNum, varNum)
diag(l) = 1
for (i in 2:varNum) {
  l[i, ] = c(rnorm(i-1), 1, rep(0, varNum-i))
}
# this (sig) is the variance-covariance matrix of the underlying dist
# in case you like to compare
sig = t(l) %*% l
u = matrix(rnorm(sampleSize*varNum), sampleSize, varNum)
x = u %*% l
## this is to insure the response is always non-negative
x = x * (x >= 0)
df1 = data.frame(x)


clustNum = 50 # param
dfCl = OrderPlotClustCentMid(df=df1, centers=clustNum, method='kmeans')
dfBu = OrderPlotClustCentMid(df=df1, centers=clustNum, method='bucket')

PlotClustResults(dfBu)
PlotClustResults(dfCl)
