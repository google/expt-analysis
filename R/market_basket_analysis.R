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



DoNotUseArrows <- function() {
  "this is just added to trigger syntax coloring based on '<- function'"
  "we use = instead of <- as it is only one character"
  "also it gives better readability to the code"
}


## Find market Basket Rules and return a data frame
FindMbRules = function(
    df, idCol, usageCol, supp=0.05, conf=0.1, orderBy="chisqPvalue") {

  df2 = df[ , c(idCol, usageCol)]
  df2 = unique(df2)
  df2 = df2[df2[ , usageCol] != "", ]
  trans = as(split(df2[ , usageCol], df2[ , idCol]), "transactions")
  rules = arules::apriori(
      trans, parameter = list(supp=supp, conf=conf, target='rules'))
  rules = sort(rules, by="lift")

  if (length(rules) == 0) {
    print("WARNING: No rules were found. WTF.")
    return( list("rules"=NULL, "rulesDf"=NULL))
  }

  rulesDf = data.frame(
      lhs = labels(lhs(rules)),
      rhs = labels(rhs(rules)),
      rules@quality)

  rulesDf[ , "rhs"] = gsub("\\}", "", gsub("\\{", "", rulesDf[ , "rhs"]))
  rulesDf[ , "lhs"] = gsub("\\}", "", gsub("\\{", "", rulesDf[ , "lhs"]))
  rulesDf[ , "lhs_size"] = sapply(
      regmatches(rulesDf[ , "lhs"],
      gregexpr(",", rulesDf[ , "lhs"])), length) + 1

  ind = nchar(rulesDf[ , "lhs"]) == 0
  rulesDf[ind, "lhs_size"] = 0
  chisq = interestMeasure(x=rules, measure="chiSquared", transactions=df2)
  chisqPvalue = pchisq(chisq, df=1, lower.tail=FALSE)

  fisherPvalue = interestMeasure(
      x=rules, measure="fishersExactTest", transactions=df2)
  rulesDf = cbind(rulesDf, chisqPvalue, fisherPvalue)

  if (!is.null(orderBy)) {
    rulesDf = rulesDf[order(rulesDf[ , orderBy]), ]
  }

  outList = list("rules"=rules, "rulesDf"=rulesDf)
  return(outList)
}

## get rules from k-tuples of products to 1
GetKto1Rules = function(
  rulesDf,
  k=NULL,
  liftThresh=1,
  suppThresh=0.00001,
  confThresh=0.01,
  lhsSizeThresh=3) {

  if (!is.null(k)) {
    rulesDf = rulesDf[rulesDf[ , 'lhs_size'] == k, ]
  }

  if (dim(rulesDf)[1] == 0) {
    return(NULL)
  }

  rulesDf = rulesDf[rulesDf[ , 'lift'] >= liftThresh, ]
  rulesDf = rulesDf[rulesDf[ , 'support'] >= suppThresh, ]
  rulesDf = rulesDf[rulesDf[ , 'confidence'] >= confThresh, ]
  rulesDf = rulesDf[rulesDf[ , 'lhs_size'] <= lhsSizeThresh, ]

  if (dim(rulesDf)[1] == 0) {
    return(NULL)
  }

  F = function(x) {
    return(paste(x, collapse=' & '))
  }

  basketDf = aggregate(
      formula=as.formula(paste('rhs', '~', 'lhs')),
      data=rulesDf[ , c('lhs', 'rhs')],
      FUN=F)
  return(basketDf)
}

## this simply gets the raw baskets and their frequency
# from transaction data
GetRawBaskets = function(df, usageCol, idCol) {
  F = function(x) {
      x = sort(x)
      out = paste(x, collapse=',')
  }
  basketDf = aggregate(
      formula=as.formula(paste(usageCol, '~', idCol)),
      data=df[ ,c(idCol, usageCol)],
      FUN=F)
  tab = table(basketDf[ , usageCol])
  tab = sort(tab, decreasing=TRUE)
  return(tab)
}
