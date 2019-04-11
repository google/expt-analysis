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

## Find market Basket Rules and return a data frame
FindMbRules = function(
    df, idCol, usageCol, supp=0.05, conf=0.1, orderBy="chisqPvalue") {

  df2 = df[ , c(idCol, usageCol)]
  df2 = unique(df2)
  df2 = df2[df2[ , usageCol] != "", ]
  trans = as(split(df2[ , usageCol], df2[ , idCol]), "transactions")
  rules = arules::apriori(
      trans, parameter = list(supp=supp, conf=conf, target="rules"))
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
    rulesDf = rulesDf[rulesDf[ , "lhs_size"] == k, ]
  }

  if (dim(rulesDf)[1] == 0) {
    return(NULL)
  }

  rulesDf = rulesDf[rulesDf[ , "lift"] >= liftThresh, ]
  rulesDf = rulesDf[rulesDf[ , "support"] >= suppThresh, ]
  rulesDf = rulesDf[rulesDf[ , "confidence"] >= confThresh, ]
  rulesDf = rulesDf[rulesDf[ , "lhs_size"] <= lhsSizeThresh, ]

  if (dim(rulesDf)[1] == 0) {
    return(NULL)
  }

  F = function(x) {
    return(paste(x, collapse=" & "))
  }

  basketDf = aggregate(
      formula=as.formula(paste("rhs", "~", "lhs")),
      data=rulesDf[ , c("lhs", "rhs")],
      FUN=F)
  return(basketDf)
}

## this simply gets the raw baskets and their frequency
# from transaction data
GetRawBaskets = function(df, usageCol, idCol) {
  F = function(x) {
      x = sort(x)
      out = paste(x, collapse=",")
  }
  basketDf = aggregate(
      formula=as.formula(paste(usageCol, "~", idCol)),
      data=df[ ,c(idCol, usageCol)],
      FUN=F)
  tab = table(basketDf[ , usageCol])
  tab = sort(tab, decreasing=TRUE)
  return(tab)
}


## this is a function to work with sequential data
# see python code (sequential_data.py) for tools to
# generate seq data
Find_sigBaskets_fromSeqData = function(
    seqDf,
    idCols=c("seq_id", "id", "date"),
    basketCol="full_seq_basket",
    liftThresh=1.5,
    suppThresh=0.00001,
    confThresh=0.5,
    lhsSizeThresh=2,
    basketSizeLowerBound=1,
    keepFreqItemOnly=FALSE,
    topItemNum=200,
    basketSep=";",
    writePath=NULL,
    fnPrefix="basket") {

  Mark(x=dim(seqDf), text="this is the uploaded seqDf shape")
  Mark(x=colnames(seqDf), text="this is the sedDf column names")
  Mark(x=seqDf[1:2, ])

  seqDf = seqDf[ , c(idCols, basketCol)]

  ## we could only keep baskets with more than one element
  # calculate basket size
  x = seqDf[ , basketCol]
  seqDf[ , "basketSize"] = lengths(regmatches(x, gregexpr(",", x))) + 1
  seqDf = seqDf[seqDf[ , "basketSize"] >= basketSizeLowerBound, ]
  Mark(x=dim(seqDf), text="dim of seqDf after removing small baskets")

  ## flatten the data
  flatDf = Flatten_RepField(df=seqDf, listCol=basketCol, sep=basketSep)
  flatDf[ , "usage"] = flatDf[ , basketCol]
  Mark(dim(flatDf), "dim(flatDf)")
  flatDf[ , "trans_id"] = do.call(
      what=function(...)paste(..., sep="-"),
      args=flatDf[ , idCols, drop=FALSE])


  df = flatDf[ , c("trans_id", "usage")]

  ## just assuring there are no repetitions
  Mark(dim(df), text="dim of flat data")
  df = unique(df)
  Mark(dim(df), text="dim of flat data, after removing reps")

  ## we could only keep top items with a given minimal freq
  usageCol = "usage"
  if (keepFreqItemOnly) {
    df[ , usageCol] = as.character(df[ , usageCol])
    tabDf = data.frame(table(df[ , usageCol]))
    tabDf = tabDf[order(tabDf[ , "Freq"], decreasing=TRUE), ]
    topUsages = tabDf[ , "Var1"][1: min(topItemNum, dim(tabDf)[1])]
    #df = df[df[ , usageCol] %in% topUsages, ]
    Mark(dim(df), text="dim of flat data, after removing rare usage categs")
  }


  out = FindMbRules(
      df=df,
      idCol="trans_id",
      usageCol="usage",
      supp=suppThresh,
      conf=confThresh,
      orderBy="lift")

  rulesDf = out[["rulesDf"]]
  Mark(dim(rulesDf), text="rulesDf shape")
  rulesDf = rulesDf[rulesDf[ , "lhs_size"] <= lhsSizeThresh, ]
  Mark(dim(rulesDf), text="this is the dim of the rules after lhs_size  filtering")
  rulesDf = rulesDf[rulesDf[ , "lift"] >= liftThresh, ]
  Mark(dim(rulesDf), text="this is the dim of the rules df after lift filtering")
  rulesDf[ , "minus_lift"] = -rulesDf[ , "lift"]
  rulesDf = rulesDf[with(rulesDf, order(lhs_size, lhs, minus_lift)), ]
  Mark(dim(rulesDf), text="this is the dim of the rules df after ordering")

  basketsKto1 = GetKto1Rules(
      rulesDf,
      k=NULL,
      liftThresh=liftThresh,
      suppThresh=suppThresh,
      confThresh=confThresh,
      lhsSizeThresh=lhsSizeThresh)

  Mark(dim(basketsKto1), "basketsKto1")

  rulesDf[ , "lhs"] =  gsub(",", ", ", rulesDf[ , "lhs"])
  rulesDf2 = rulesDf[ , c(
      "lhs", "rhs", "lhs_size", "support", "confidence", "lift")]
  names(rulesDf2) = c(
      "rule_left_hand_side", "rule_right_hand_side", "rule_lhs_size",
      "support", "confidence", "lift")

  if (!is.null(writePath)) {
    fn = paste0(writePath, fnPrefix,  "_rules", ".csv")
    Mark(fn, text="this file is being written")
    fn = file(fn)
    write.csv(file=fn, x=SignifDf(rulesDf2, 3), row.names=FALSE)

    fn = paste0(writePath, fnPrefix, "_rules_summary", ".csv")
    Mark(fn, text="this file is being written")
    fn = file(fn)
    write.csv(file=fn, x=SignifDf(basketsKto1, 3), row.names=FALSE)
  }

  return(list(
      "rulesDf"=SignifDf(rulesDf, 3),
      "rulesDfSummary"=SignifDf(basketsKto1, 3)))

}
