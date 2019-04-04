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

SimData = function() {

  ss = 1000
  bucketNum = 20

  id = 1:ss
  bucket = id %% bucketNum
  x1 = runif(ss, -0.5, 0.5)
  x2 = runif(ss, -0.5, 0.5)
  x3 = runif(ss, -0.5, 0.5)
  expt_numeric = x3 > 0.5
  expt = rep("cont", ss)
  expt[expt_numeric] = "treat"
  y = 0.1 + 2*x1 + -2*x2 + 0.1*expt_numeric + 0.1*runif(ss, -0.5, 0.5)

  df = data.frame(id, bucket, x1, x2, x3, expt, y)
  dt = data.table(df)

  return(dt)

}

AddDichomVar = function(dt, col, num=6) {

  x = dt[ , get(col)]
  x = na.omit(x)
  step = 1 / num
  qs = quantile(x, seq(step, 1-step, step))
  qs = c(-Inf, qs, Inf)
  dt[ , paste0(col, "_categ")] = cut(x, qs)
  return(dt)

}


FitAggModel = function(dt, yCol, predCols) {

  dt = data.table(dt)
  Mean = function(x) {
    mean(x, na.rm=TRUE)
  }

  dtAgg = dt[ , Mean(get(yCol)), by=mget(predCols)]
  colnames(dtAgg) = c(predCols, yCol)
  return(dtAgg)

}


PredAggModel = function(newDt, dtAgg, predCols) {

  dt = merge(newDt, dtAgg, by=predCols, all.x=TRUE, all=FALSE)
  return(dt)

}


AssessMod = function(y, yPred) {

  d = na.omit(y - yPred)
  rmse = sqrt(sum(d^2) / length(d))
  corr = cor(y, yPred, use="pairwise.complete.obs")
  covar = cor(y, yPred, use="pairwise.complete.obs")
  r2 = 1 - (rmse^2) / var(y)


}
