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

# Functions for expt analysis

LogitInv = function(x) {
  exp(x) / (1 + exp(x))
}

Logit = function(x) {
  log(x / (1-x))
}

## generates random timestamps
SimTimeStamps = function(
    n,
    timestamp1,
    timestamp2) {
  # generates random timestamps between two given dates
  st = as.POSIXct(as.Date(timestamp1))
  et = as.POSIXct(as.Date(timestamp2))
  dt = as.numeric(difftime(timestamp2, timestamp1, unit="sec"))
  ev = sort(runif(n, 0, dt))
  rt = st + ev
  return(rt)
}

# generates a population of users
SimUserAttr = function(
    userNum,
    distbns=list(
        "country"=list("US"=0.2, "IN"=0.4, "BR"=0.2, "JP"=0.1, "FR"=0.1),
        "gender"=list("MALE"=0.5, "FEMALE"=0.5),
        "expt_id"=list("cont"=0.5, "treat"=0.5)),
    balanceCol="expt_id") {
  # generates a user attributes dataframe
  #with given labels and weights

 df = data.frame("user_id"=1:userNum)

  for (col in names(distbns)) {
    dist = distbns[[col]]
    df[ , col] = sample(
    names(dist),
    userNum,
    replace=TRUE,
    prob=unlist(dist, use.names=FALSE))
  }

  if (!is.null(balanceCol)) {
    res = BalanceSampleSize(
      df=df,
      sliceCols=balanceCol,
      itemCols="user_id",
      sliceCombinValues_toBalance=NULL)

    df = res[["subDf"]]
  }

  return(df)
}

## adding counter-factuals to user data
AddCounterfactual = function(
    df, exptCol, switchPair=NULL, newColName="cfactual") {

  df = as.data.frame(df)
  if (is.null(switchPair)) {
    switchPair = unique(df[ , exptCol])
  }

  if (length(switchPair) > 2) {
    print("switchPair cannot have more than two elements.")
    return()
  }

  df[ , "cfactual"] = "factual"
  x = df[ , exptCol]
  y = x
  y[x == switchPair[1]] = switchPair[2]
  y[x == switchPair[2]] = switchPair[1]
  df2 = df
  df2[ , exptCol] = y
  df2[ , "cfactual"] = "cfactual"

  return(list("fullDf"=rbind(df, df2), "obsDf"=df, "cfDf"=df2))
}

# generates a linear combination of the covariate effects
GenReg_linearPred = function(
    df,
    mu0,
    covars,
    valueCol) {
  # generates a linear preds: eta=X beta
  # with given labels and weights
  df[ , valueCol] = mu0
  for (var in names(covars)) {
    coefs = covars[[var]]
    for (label in names(coefs)) {
      df[ , valueCol] = df[ , valueCol] + (df[ , var] == label) * coefs[[label]]
    }
  }
  return(df)
}

# adds user random effects
# we allow for repetitions in user dataframe
# in this way we could have counterfactuals
AddUser_randomEffects = function(
    userDf,
    userCol,
    valueCols,
    userVarCov_chol,
    userVarCov=NULL) {

  if (is.null(userVarCov)) {
    userVarCov =  userVarCov_chol %*% t(userVarCov_chol)
  }

  users = unique(userDf[ , userCol])
  userOnlyDf = data.frame("V1"=users)
  names(userOnlyDf) = userCol

  userNum = length(users)
  userRe = mvrnorm(userNum, mu=rep(0, length(valueCols)), Sigma=userVarCov)
  userRe = data.frame(userRe)
  names(userRe) = paste0(valueCols, "_userRandomEffect")
  userRe[ , userCol] = userOnlyDf[ , userCol]

  userDf2 = merge(userDf, userRe, by="user_id", all.x=TRUE)

  for (i in 1:length(valueCols)) {
    col = valueCols[i]
    userDf2[ , col] = (
        userDf2[ , col] + userDf2[ , paste0(col, "_userRandomEffect")])
  }

  ## drop the extra columns
  userDf2 = userDf2[ , !(names(userDf2) %in% paste0(valueCols, "_userRandomEffect"))]
  return(userDf2)
}

# transforms the linear combinations to conditional expectations
# via link functions
TransfCols = function(
    df,
    transfList) {
  cols = names(transfList)
  for (col in cols) {
    if (!is.null(transfList)) {
      df[col] = transfList[[col]](df[col])
    }
  }
  return(df)
}

# generates usage data
GenUsageDf = function(
    userDf,
    valueDistbns,
    timestamp1,
    timestamp2,
    CountSim=function(n, lambda){rpois(n, lambda)},
    noTimestamps=FALSE,
    parallel=FALSE,
    parallel_outfile="") {

  valueCols = names(valueDistbns)

  userDf[ , "obs_count_for_user"] = CountSim(
      n=nrow(userDf), lambda=userDf[ , "count"])

  GenUsage_perRow = function(i) {
    df0 = userDf[i, , drop=FALSE]
    eventCount = df0[1, "obs_count_for_user"]
    if (eventCount == 0) {
      return(NULL)
    }

    timestamps = rep(NA, eventCount)
    if (!noTimestamps) {
      timestamps = SimTimeStamps(
          n=eventCount, timestamp1=timestamp1, timestamp2=timestamp2)
    }
    df1 = data.frame("timestamp"=timestamps)

    for (col in valueCols) {
      valueMean = as.numeric(df0[1, col])
      values = valueDistbns[[col]](n=eventCount, x=valueMean)
      df1[ , paste0("obs_", col)] = values
    }
    df2 = merge(df0, df1, how="outer")
    return(df2)
  }

  if (!parallel) {
    outDf = lapply(X=1:nrow(userDf), FUN=GenUsage_perRow)
  } else {
    suppressMessages(library("parallel"))
    closeAllConnections()
    no_cores = detectCores() - 3
    no_cores = min(no_cores, nrow(userDf) + 1)
    Mark(no_cores, "no_cores")
    # Initiate cluster
    cl = makeCluster(no_cores, outfile=parallel_outfile)
    clusterExport(
            cl=cl,
            list(
                "userDf", "valueDistbns", "timestamp1", "timestamp2",
                "noTimestamps", "valueCols", "GenUsage_perRow",
                "Src"),
            envir=environment())
    estimList =  parLapply(cl=cl, X=1:nrow(userDf), fun=GenUsage_perRow)
    stopCluster(cl)
    closeAllConnections()
  }

  x = do.call(rbind, lapply(1:nrow(userDf), FUN=GenUsage_perRow))
  return(x)
}

## generates a function which assigns "cookie buckets" to user_ids
GenModFcn = function(modNum) {
  F = function(id) {
    return(id %% modNum)
  }
  return(F)
}

## this function create a wide format corresponding to the given pair
# e.g. for the (expt, cont) pair
# from the wide format, it then calculates a diff between the specified pair
DiffDf = function(
    df,
    compareCol,
    valueCols,
    itemCols=NULL,
    comparePair=NULL,
    Diff=NULL,
    diffFcnList=NULL) {

  ## if Diff function is not given, it will default to minus
  if (is.null(Diff) & is.null(diffFcnList)) {
    Diff = function(x, y)(x - y)
  }

  if (is.null(itemCols)) {
    itemCols = "item"
    df[ , "item"] = 1
  }

  dt = data.table(df)
  if (is.null(comparePair)) {
    comparePair = names(table(dt[ , get(compareCol)])[1:2])
  }

  dt = dt[get(compareCol) %in% comparePair]

  formulaStr = paste0(itemCols, "~", compareCol)

  ## check if there are repetitions in compareCol + itemCols
  dt2 = dt[ , mget(c(itemCols, compareCol))]

  if (nrow(dt2) > nrow(unique(dt2))) {
    print("WARNING: itemCols + compareCols not unique. dcast would fail.")
    return(NULL)
  }

  wideDt = dcast(data=dt, formula=as.formula(formulaStr), value.var=valueCols)

  for (i in 1:length(valueCols)) {
    col = valueCols[i]
    diffCol = paste0(col, "_", comparePair[1], "_vs_", comparePair[2])
    col1 = paste0(col, "_", comparePair[1])
    col2 = paste0(col, "_", comparePair[2])
    G = Diff
    if (!is.null(diffFcnList)) {
      G = diffFcnList[[i]]
    }
    wideDt[ , diffCol] = G(wideDt[ , get(col1)], wideDt[ , get(col2)])
  }

  return(wideDt)
}


TestDiffDf = function() {

  n = 1000
  x1 = abs(rnorm(n))
  x2 = abs(rnorm(n))
  y = abs(2*x1 + 3*x2 + rnorm(n))
  df = data.frame(x1, x2, y)
  df[ , "user_id"] = 1:1000
  df[ , "expt_id"] = sample(c("treat", "cont"), n, replace=TRUE)
  df[ , "country"] = sample(c("us", "japan"), n, replace=TRUE)

  dt = data.table(df)

  dtAgg = DtSimpleAgg(
      dt=dt, gbCols=c("expt_id", "country"), valueCols=c("x1", "x2", "y"),
      cols=NULL, F=sum)

  DiffDf(
      df=dtAgg, compareCol="expt_id", itemCols="country",
      valueCols=c("x1", "x2", "y"),
      comparePair=c("treat", "cont"), Diff=NULL, diffFcnList=NULL)
}


CalcDiffMetrics = function(
    df, compareCol, valueCols, AggF, comparePair, itemCols=NULL, Diff=NULL) {

  dt = data.table(df)

  dtAgg = DtSimpleAgg(
      dt=dt, gbCols=c(compareCol, itemCols),
      valueCols=valueCols,
      cols=NULL, F=AggF)

  out = DiffDf(
      df=dtAgg, compareCol=compareCol, itemCols=itemCols,
      valueCols=valueCols, comparePair=comparePair,
      Diff=Diff, diffFcnList=NULL)

  return(out)
}

TestCalcDiffMetrics = function() {

  n = 1000
  x1 = abs(rnorm(n))
  x2 = abs(rnorm(n))
  y = abs(2*x1 + 3*x2 + rnorm(n))
  df = data.frame(x1, x2, y)
  df[ , "user_id"] = 1:1000
  df[ , "expt_id"] = sample(c("treat", "cont"), n, replace=TRUE)
  df[ , "country"] = sample(c("us", "japan"), n, replace=TRUE)

  dt = data.table(df)

  CalcDiffMetrics(
      df, compareCol="expt_id", valueCols=c("x1", "x2", "y"), AggF=sum,
      comparePair=c("treat", "cont"), itemCols="country",
      Diff=function(x, y) {(x - y) / abs(x)})
}


GenericDiffMetrics = function(
    df, compareCol, valueCols, comparePair, itemCols=NULL) {

  Sum = function(x) {
    sum(as.double(x))
  }

  aggDiffList = list(
      "mean_ratio"=list("AggF"=mean, "Diff"=function(x, y) {x / y}),
      "sum_ratio"=list("AggF"=Sum, "Diff"=function(x, y) {x / y}),
      "mean_perc_change"=list("AggF"=mean, "Diff"=function(x, y) {100 * (x-y) / y}),
      "sum_perc_change"=list("AggF"=Sum, "Diff"=function(x, y) {100 * (x-y) / y}))


  F = function(name) {
    aggDiffPair = aggDiffList[[name]]
    out = CalcDiffMetrics(
        df=df, compareCol=compareCol, valueCols=valueCols,
        AggF=aggDiffPair[["AggF"]],
        comparePair=comparePair, itemCols=itemCols,
        Diff=aggDiffPair[["Diff"]])
    out[ , "metric"] = name
    return(out)
  }


  diffDfList = lapply(X=names(aggDiffList), FUN=F)

  diffDf = do.call(what=rbind, args=diffDfList)

  return(diffDf)
}


TestGenericDiffMetrics = function() {

  n = 1000
  x1 = abs(rnorm(n))
  x2 = abs(rnorm(n))
  y = abs(2*x1 + 3*x2 + rnorm(n))
  df = data.frame(x1, x2, y)
  df[ , "user_id"] = 1:1000
  df[ , "expt_id"] = sample(c("treat", "cont"), n, replace=TRUE)
  df[ , "country"] = sample(c("us", "japan"), n, replace=TRUE)

  GenericDiffMetrics(
      df=df, compareCol="expt_id", valueCols=c("x1", "x2"),
      comparePair=c("treat", "cont"), itemCols="country")
}


## fitting and prediction with cross-validated errors
FitPred = function(
    df, newDf, valueCol, predCols, family, predQuantLimits=c(0, 1)) {

  formulaStr = paste0(valueCol, "~",  paste(predCols, collapse="+"))
  formula = as.formula(formulaStr)

  ## with adjust we limit the predicted values to observed quantiles
  Adjust = function(x)x
  if (!is.null(predQuantLimits)) {
    yMin = quantile(df[ , valueCol], predQuantLimits[1])
    yMax = quantile(df[ , valueCol], predQuantLimits[2])
    Adjust = function(x) {
      x = pmax(x, yMin)
      x = pmin(x, yMax)
      return(x)
    }
  }

  F = function(df, newDf) {
    mod = glm(formula=formula, family=family, data=df)
    yFit = mod[["fitted.values"]]
    yFit = Adjust(yFit)
    fitted_cor = cor(mod[["y"]], yFit)
    yPred = predict(object=mod, newdata=newDf)
    yPred = Adjust(yPred)
    newDf[ , valueCol] = yPred
    return(list(
        "mod"=mod, "fitted_cor"=fitted_cor, "newDf"=newDf, "yPred"=yPred))
  }

  n = nrow(df)
  varY = var(df[ , valueCol])
  res = F(df=df, newDf=df)
  varFit = var(res[["yPred"]])
  cov_y_fit = cov(df[ , valueCol], res[["yPred"]])
  k = round(n*0.8)
  samp1 = sample(1:n, k)
  samp2 = setdiff((1:n), samp1)

  trainDf = df[samp1, ]
  testDf = df[samp2, ]
  yPred = F(df=trainDf, newDf=testDf)[["yPred"]]
  varTest = var(yPred)

  cv_cor = cor(testDf[ , valueCol], yPred)
  cv_cov = cov(testDf[ , valueCol], yPred)
  cv_mse = mean(na.omit(testDf[ , valueCol] - yPred)^2)
  cv_mae = mean(na.omit(abs(testDf[ , valueCol] - yPred)))

  res = F(df=df, newDf=newDf)

  res[["cv_cor"]] = cv_cor
  res[["cv_mse"]] = cv_mse
  res[["cv_mae"]] = cv_mae
  res[["var_y"]] = varY
  res[["theta"]] = cv_cor * sqrt(varY / varFit)
  res[["theta2"]] = cov_y_fit / varFit
  res[["theta3"]] = cv_cov / varTest
  res[["theta"]] = max(min(res[["theta"]], 1.0), 0)
  res[["sd_ratio"]] = sqrt(1 - cv_cor^2)

  return(res)
}

TestFitPred = function() {
  n = 1000
  x1 = rnorm(n)
  x2 = rnorm(n)
  y = 2*x1 + 3*x2 + rnorm(n)
  df = data.frame(x1, x2, y)

  trainDf = df[1:n/2, ]
  testDf = df[(n/2 + 1):100, ]

  res = FitPred(
      df=trainDf, newDf=testDf, valueCol="y", predCols=c("x1", "x2"),
      family=gaussian)

  res[!(names(res) %in% c("newDf", "yPred"))]

  plot(res[["yPred"]], testDf[ , "y"])
}

## fit and predicts for multiple value columns
FitPred_multi = function(
    df, newDf, valueCols, predCols, familyList=NULL, commonFamily=gaussian,
    predQuantLimits=c(0, 1)) {

  infoDf = setNames(
      data.frame(matrix(ncol=9, nrow=0)),
      c("valueCol", "formulaStr", "fitted_cor", "cv_cor",
      "cv_mse", "cv_mae", "var_y", "theta", "sd_ratio"))

  modList = list()
  for (valueCol in valueCols) {
    formulaStr = paste0(valueCol, "~",  paste(predCols, collapse="+"))
    formula = as.formula(formulaStr)
    if (!is.null(familyList)) {
      family = familyList[[valueCol]]
    } else {
      family = commonFamily
    }

    res = FitPred(
        df=df, newDf=newDf, valueCol=valueCol, predCols=predCols, family=family)
    newDf = res[["newDf"]]

   infoDf[nrow(infoDf) + 1, 1:2] = c(valueCol, formulaStr)
   infoDf[nrow(infoDf), 3:9] = c(
       res[["fitted_cor"]], res[["cv_cor"]],
       res[["cv_mse"]], res[["cv_mae"]], res[["var_y"]], res[["theta"]],
       res[["sd_ratio"]])

    modList[[valueCol]] = res[["mod"]]
  }

  for (col in c("fitted_cor", "cv_cor",
                "cv_mse", "cv_mae", "var_y",
                "theta", "sd_ratio")) {
    infoDf[ , col] = as.numeric(infoDf[ , col])
  }

  df[ , "obs_pred"] = "obs"
  newDf[ , "obs_pred"] = "pred"
  concatDf = rbind(df, newDf)

  return(list(
      "newDf"=newDf,
      "concatDf"=concatDf,
      "infoDf"=infoDf,
      "modList"=modList))
}

## creates a regression table with estimates and p-values
# it does that in two ways, once with and once without predCols if given
RegTab_exptPredCols = function(
    dt, valueCols, predCols=NULL, writePath,
    family=gaussian, ss=NULL, signif=3) {

  dt2 = copy(dt)
  if (!is.null(dt)) {
    dt2 = dt[sample(.N, ss)]
  }

  df = data.frame(dt2)

  res = FitPred_multi(
      df=df,
      newDf=df,
      valueCols=valueCols,
      predCols=c("expt_id"),
      commonFamily=family,
      predQuantLimits=c(0, 1))

  res[["infoDf"]][ , c("valueCol", "cv_cor", "theta", "sd_ratio")]
  #cat("\n\n")
  modList = res[["modList"]]
  regTab = RegModList_coefTableSumm(modList=modList, keepVars="expt_idtreat")
  regTab[ , "ss"] = ss
  regTab[ , "complexity"] = "w/o pred cols"

  regTabList[["wo_predCols"]] = regTab

  #@title regression with vars and expt_id
  if (!is.null(predCols)) {
    res = FitPred_multi(
        df=df,
        newDf=df,
        valueCols=valueCols,
        predCols=c(predCols, "expt_id"),
        commonFamily=family,
        predQuantLimits=c(0, 1))

    res[["infoDf"]][ , c("valueCol", "cv_cor", "theta", "sd_ratio")]
    cat("\n\n")
    modList = res[["modList"]]
    regTab =  RegModList_coefTableSumm(modList=modList, keepVars="expt_idtreat")
    regTab[ , "ss"] = ss
    regTab[ , "complexity"] = "w pred cols"
    regTabList[["w_predCols"]] = regTab
  }


  regTab_all = do.call(what=rbind, args=regTabList)
  regTab_all = StarPvalueDf(regTab_all)
  regTab_all = regTab_all[order(regTab_all[ , "model_label"]), ]

  fn0 = paste0(writePath, "reg_tab_", fnSuffix, ".csv")
  fn0 = tolower(fn0)
  print(fn0)
  fn = file(fn0, "w")
  write.csv(x=regTab_all, file=fn)
  close(fn)

  regTab_all = DfSubsetCols(regTab_all, dropCols=c("ss"))
  regTab_all[ , "var"] = "treat"
  rownames(regTab_all) = NULL


  fn0 = paste0("reg_tab_", fnSuffix)
  caption = gsub(x=fn0, "_", " ")
  label = fn0
  fn0 = paste0(writePath, fn0, ".tex")
  fn0 = tolower(fn0)
  print(fn0)
  fn = file(fn0, "w")
  x = xtable2(regTab_all, caption=caption, label=label, digit=signif)
  print(
       x=x, file=fn, include.rownames=FALSE,
       hline.after=c(-1, 0, 1:nrow(regTab_all)),
       size="tiny")
  close(fn)

  return(list("regTab"=regTab_all, latexTab=x))
}

TestFitPred_multi = function() {

  n = 1000
  x1 = rnorm(n)
  x2 = rnorm(n)
  y1 = 2*x1 + 3*x2 + rnorm(n)
  y2 = 2*x1 + 3*x2 + rnorm(n, sd=5)
  y3 = 2*x1 + 3*x2 + rnorm(n, sd=10)

  df = data.frame(x1, x2, y1, y2, y3)

  trainDf = df[1:n/2, ]
  testDf = df[(n/2 + 1):100, ]

  res = FitPred_multi(
      df=trainDf,
      newDf=testDf,
      valueCols=c("y1", "y2", "y3"),
      predCols=c("x1", "x2"),
      familyList=NULL,
      commonFamily=gaussian)

  Mark(res[["infoDf"]])
  plot(res[["newDf"]][ , "y1"], testDf[ , "y1"])
}

## calculates err for difference between two data sets on valueCols
# note that R2 is also included and its assymetric
CompareContiVars = function(df1, df2, valueCols) {

  infoDf = setNames(
      data.frame(matrix(ncol=6, nrow=0)),
      c("valueCol", "cor", "rmse", "mae", "perc_err", "R2"))

  for (col in valueCols) {
    cor = cor(df1[ , col], df2[ , col])
    rmse = sqrt(mean((df1[ , col] - df2[ , col])^2))
    mae = mean(abs(df1[ , col] - df2[ , col]))
    b = mean(abs(df1[ , col]))/2 + mean(abs(df1[ , col]))/2
    perc_err = mae / b
    r2 = 1 - (mean((df1[ , col] - df2[ , col])^2) / var(df1[ , col]))

    infoDf[nrow(infoDf) + 1, ] = c(col, cor, rmse, mae, perc_err, r2)
  }

  for (col in c("cor", "rmse", "mae", "perc_err", "R2")) {
    infoDf[ , col] = as.numeric(infoDf[ , col])
  }

  return(infoDf)
}

TestCompareContiVars = function() {
  n = 100
  x1 = rnorm(n)
  x2 = rnorm(n)
  y1 = 2*x1 + 3*x2 + rnorm(n)
  y2 = 2*x1 + 3*x2 + rnorm(n, sd=5)

  df1 = data.frame(x1, x2, y1, y2)

  y1 = 2*x1 + 3*x2 + rnorm(n)
  y2 = -2*x1 + 3*x2 + rnorm(n, sd=5)

  df2 = data.frame(x1, x2, y1, y2)

  CompareContiVars(df1=df1, df2=df2, valueCols=c("y1", "y2"))
}

## balancing the sample sizes (in terms of number of items)
# we assign the minimum available sample size to all slices
# this is done by defining a new column: which is isBalancedSample
# if you only like to do partial balancing on some slice Values
# specify those slice values in sliceCombinValues_toBalance
BalanceSampleSize = function(
    df,
    sliceCols,
    itemCols=NULL,
    sliceCombinValues_toBalance=NULL) {

  if (is.null(itemCols)) {
    itemCols = c("dummy_item")
    df[ , "dummy_item"] = 1:nrow(df)}

  df = Concat_stringColsDf(
    df=df,
    cols=itemCols,
    colName="item_combin",
    sepStr="-")

  df = Concat_stringColsDf(
    df=df,
    cols=sliceCols,
    colName="slice_combin",
    sepStr="-")

  itemColsStr = paste(itemCols, collapse="_")
  sliceColsStr = paste(sliceCols, collapse="_")

  df2 = unique(df[ , c("item_combin", "slice_combin")])
  df3 = df2[order(df2[ , "slice_combin"]), ]
  dt3 = data.table(df3)

  dfItemCount_perSlice = data.frame(dt3[ , .N, by="slice_combin"])
  names(dfItemCount_perSlice)[names(dfItemCount_perSlice) == "N"] = "item_combin_count"

  dfItemCount_perSlice$slice_item_index =  lapply(
      X=as.list(dfItemCount_perSlice[ , "item_combin_count"]),
      FUN=function(x){list(1:x)})

  if (is.null(sliceCombinValues_toBalance)) {
    minSs = min(dfItemCount_perSlice[ , "item_combin_count"])
    # if there is only once slice remaining, we assign False to all
    if (nrow(dfItemCount_perSlice) < 2){
      minSs = -Inf
    }
  }  else {
    df0 = dfItemCount_perSlice[
      dfItemCount_perSlice[ , "slice_combin"] %in% sliceCombinValues_toBalance, ]
    minSs =  min(df0[ , "item_combin_count"])
    # if there is only once slice remaining, we assign False to all
    if (nrow(df0) < 2){
      minSs = -Inf
    }
  }

  dfItemSliceIndex = Flatten_RepField(
      df=dfItemCount_perSlice,
      listCol="slice_item_index")

  dfItemSliceIndex = DropCols(
      df=dfItemSliceIndex,
      cols="item_combin_count")

  colName = paste0(sliceColsStr, ".", itemColsStr, "_index")
  boolColName = paste0("balanced_", sliceColsStr, "__", itemColsStr)

  setnames(x=dfItemSliceIndex, old="slice_item_index", new=colName)

  if (is.null(sliceCombinValues_toBalance)) {
    dfItemSliceIndex[boolColName] = dfItemSliceIndex[colName] <= minSs
  } else {
    dfItemSliceIndex[ , boolColName] = (
        (dfItemSliceIndex[ , colName] <= minSs) |
        !dfItemSliceIndex[ , "slice_combin"] %in% sliceCombinValues_toBalance)
  }

  df3[ , colName] = dfItemSliceIndex[ , colName]
  df3[ , boolColName] = dfItemSliceIndex[ , boolColName]
  fullDf = merge(df, df3, all.x=TRUE, by=c("item_combin", "slice_combin"))

  df0 = fullDf[ , c(sliceCols, boolColName, "item_combin")]
  dt0 = data.table(df0)
  infoDf = dt0[ , .(item_combin_count=length(unique(item_combin))),
               by=c(sliceCols, boolColName)]

  fullDf = DropCols(df=fullDf, cols=c("item_combin", "slice_combin"))
  subDf = fullDf[fullDf[ , boolColName], ]
  subDf = DropCols(df=subDf, cols=c(colName, boolColName))
  return(list("fullDf"=fullDf, "subDf"=subDf, "infoDf"=infoDf))
}

TestBalanceSampleSize = function() {

  df = SimUsageDf_e2e(userNum=50)[["userDf"]]
  Mark(df[1:2, ])

  res = BalanceSampleSize(
      df=df,
      sliceCols=c("country"),
      itemCols=c("user_id"),
      sliceCombinValues_toBalance=NULL)

  Mark(res["infoDf"])

  ## partial balancing
  res = BalanceSampleSize(
      df=df,
      sliceCols="country",
      itemCols=c("user_id", "date"),
      sliceCombinValues_toBalance=c("JP", "FR"))

  Mark(res["infoDf"])
}

## add predictions
# this adds two types of predictions
# one is with control data only
# one is with all data and using expt_id as a predictor
# predictions are added for factual arms
# as well as counterfactual arms,
# in which we assign a user to the opposite arm
AddPred_toUserData = function(
    userDt_fromUsage_obs,
    predCols,
    valueCols) {

  ## augment the data with counterfactual data
  # so for each unit a counterfactual unit with opposite expt label is added
  # the label: cf tells us which unit is factual and which is cf
  res = AddCounterfactual(
      df=userDt_fromUsage_obs, exptCol="expt_id")

  userDf_fromUsage_fac = res[["obsDf"]]
  userDf_fromUsage_cf = res[["cfDf"]]

  # we remove the values from the cf data since the values are not obs
  for (col in valueCols) {
    userDf_fromUsage_cf[ , col] = NA
  }

  userDf_fromUsage_withCf = rbind(userDf_fromUsage_fac, userDf_fromUsage_cf)
  userDf_fromUsage_fac_contOnly = (
      userDf_fromUsage_fac[userDf_fromUsage_fac[ , "expt_id"] == "cont" , ])

  ## we create a df ready for adding predictions
  # this includes the fac and cf data
  userDf_fromUsage_modelPred = userDf_fromUsage_withCf

  # we reset all the values to NA so that model preds can be filled
  for (col in valueCols) {
    userDf_fromUsage_modelPred[ , col] = NA
  }


  fitRes_allDataNoExptId = FitPred_multi(
      df=userDf_fromUsage_fac,
      newDf=userDf_fromUsage_modelPred,
      valueCols=valueCols,
      predCols=c(predCols),
      #predCols=c(predCols, "expt_id", paste0(predCols, "*", "expt_id"))
      familyList=NULL,
      commonFamily=gaussian)

  fitRes_withTreatData = FitPred_multi(
      df=userDf_fromUsage_fac,
      newDf=userDf_fromUsage_modelPred,
      valueCols=valueCols,
      predCols=c(predCols, "expt_id"),
      #predCols=c(predCols, "expt_id", paste0(predCols, "*", "expt_id"))
      familyList=NULL,
      commonFamily=gaussian)

  ## fitted only using control data and sliceCols
  fitRes_contDataOnly = FitPred_multi(
      df=userDf_fromUsage_fac_contOnly,
      newDf=userDf_fromUsage_modelPred,
      valueCols=valueCols,
      predCols=predCols,
      familyList=NULL,
      commonFamily=gaussian)

  predDf_contDataOnly = fitRes_contDataOnly[["newDf"]]
  predDf_withTreatData = fitRes_withTreatData[["newDf"]]
  predDf_allDataNoExptId = fitRes_allDataNoExptId[["newDf"]]

  # subset the predictions on factual for validation purposes
  # note we do not have data on cf
  predDf_contDataOnly_fac = (
      predDf_contDataOnly[predDf_contDataOnly[ , "cfactual"] == "factual", ])

  predDf_withTreatData_fac = (
      predDf_withTreatData[predDf_withTreatData[ , "cfactual"] == "factual", ])

  predDf_allDataNoExptId_fac = (
      predDf_allDataNoExptId[predDf_allDataNoExptId[ , "cfactual"] == "factual", ])


  # calc err between observed and model pred
  errDf_contDataOnly = CompareContiVars(
      df1=userDf_fromUsage_fac,
      df2=predDf_contDataOnly_fac,
      valueCols=valueCols)

  # calc err between observed and model pred
  errDf_withTreatData = CompareContiVars(
      df1=userDf_fromUsage_fac,
      df2=predDf_withTreatData_fac,
      valueCols=valueCols)

  # calc err between observed and model pred
  errDf_allDataNoExptId = CompareContiVars(
      df1=userDf_fromUsage_fac,
      df2=predDf_allDataNoExptId_fac,
      valueCols=valueCols)

  return(list(
    "fitRes_contDataOnly"=fitRes_contDataOnly,
    "fitRes_withTreatData"=fitRes_withTreatData,
    "fitRes_allDataNoExptId"=fitRes_allDataNoExptId,
    "errDf_contDataOnly"=errDf_contDataOnly,
    "errDf_withTreatData"=errDf_withTreatData,
    "errDf_allDataNoExptId"=errDf_allDataNoExptId,
    "predDf_contDataOnly"=predDf_contDataOnly,
    "predDf_withTreatData"=predDf_withTreatData,
    "predDf_allDataNoExptId"=predDf_allDataNoExptId))
}

## calculate pred based means (averaged across users)
# for treat, control and their counterfactuals
# we also calculate n_t: user num on treatment
# and n_c: user_num on control
# these could be then used for calculating adjustments
PredBased_userLevelMeans = function(
    userDt_fromUsage_obs, valueCols, predCols) {

  #userCntDt = DtSimpleAgg(
  #    dt=userDt_fromUsage_obs,
  #    gbCols="expt_id",
  #    valueCols="user_id",
  #    F=function(x)length(unique(x)))

  #n_t = userCntDt[expt_id == "treat", user_id]
  #n_c = userCntDt[expt_id == "cont", user_id]

  predRes = AddPred_toUserData(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      predCols=predCols,
      valueCols=valueCols)

  #fitRes_contDataOnly = predRes[["fitRes_contDataOnly"]]
  #fitRes_withTreatData = predRes[["fitRes_withTreatData"]]

  #errDf_contDataOnly = predRes[["errDf_contDataOnly"]]
  #errDf_withTreatData = predRes[["errDf_withTreatData"]]

  predDf_contDataOnly = predRes[["predDf_contDataOnly"]]
  predDf_withTreatData = predRes[["predDf_withTreatData"]]
  predDf_allDataNoExptId = predRes[["predDf_allDataNoExptId"]]
  predDt_contDataOnly = data.table(predDf_contDataOnly)
  predDt_withTreatData = data.table(predDf_withTreatData)
  predDt_allDataNoExptId = data.table(predDf_allDataNoExptId)

  F = function(predDt) {
    # calculate means
    aggDt = DtSimpleAgg(
        dt=predDt,
        gbCols=c("expt_id", "cfactual"),
        valueCols=valueCols,
        F=mean)

    aggDt[ , "expt_id_cfactual"] = paste(
        aggDt[ , expt_id],
        aggDt[ , cfactual],
        sep="_")

    return(aggDt)
  }

  aggDt_contDataOnly = F(predDt=predDt_contDataOnly)
  aggDt_withTreatData = F(predDt=predDt_withTreatData)
  aggDt_allDataNoExptId = F(predDt=predDt_allDataNoExptId)

  return(list(
      "userLevMeans_contDataOnly"=aggDt_contDataOnly,
      "userLevMeans_withTreatData"=aggDt_withTreatData,
      "userLevMeans_allDataNoExptId"=aggDt_allDataNoExptId,
      "infoDf_contDataOnly"=predRes[["fitRes_contDataOnly"]][["infoDf"]],
      "infoDf_withTreatData"=predRes[["fitRes_withTreatData"]][["infoDf"]],
      "infoDf_allDataNoExptId"=predRes[["fitRes_allDataNoExptId"]][["infoDf"]]))
}

## assess pred power and calculate theta
AssessPredPower_calcTheta = function(
    userDt, valueCols, predCols,
    writeIt=FALSE, writePath="",
    fnSuffix="",
    latexKeepCols=c("formulaStr", "cv_cor", "theta", "sd_ratio"),
    latexRenameCols=c("model_formula", "cv_cor", "theta", "sd_ratio")) {

  res = PredBased_userLevelMeans(
     userDt_fromUsage_obs=userDt,
     valueCols=valueCols,
     predCols=predCols)

  res = lapply(
      FUN=function(x){RoundDf(x, 3)},
      X=res[c("infoDf_contDataOnly", "infoDf_withTreatData",
              "infoDf_allDataNoExptId")])

  if (writeIt) {
    for (name in names(res)) {
      fn0 = paste0(writePath, "pred_accuracy_", name, "_", fnSuffix, ".csv")
      fn0 = tolower(fn0)
      print(fn0)
      fn = file(fn0, "w")
      write.csv(x=res[[name]], file=fn)
      close(fn)

      # lets write the tex file too
      fn0 = paste0("pred_accuracy_", name, "_", fnSuffix)
      caption = gsub(x=fn0, "_", " ")
      label = fn0
      fn0 = paste0(writePath, fn0, ".tex")
      fn0 = tolower(fn0)
      print(fn0)
      fn = file(fn0, "w")
      df = res[[name]][ , latexKeepCols]
      names(df) = latexRenameCols
      if ("model_formula" %in% names(df)) {
        df[ , "model_formula"] = gsub(x=df[ , "model_formula"], "~", ": ")
      }
      x = xtable2(df, caption=caption, label=label, include.rownames=FALSE)
      print(x=x, file=fn)
      close(fn)
    }
  }

  return(res)
}

## This is for debugging only
# we can investigate if the adjustments are on average neutral
# close to zero or one depending on diff
PredBased_Diffs = function(
   userDt_fromUsage_obs,
   valueCols,
   predCols,
   comparePair,
   Diff) {

  modelPred_data = PredBased_userLevelMeans(
    userDt_fromUsage_obs=userDt_fromUsage_obs,
    valueCols=valueCols,
    predCols=predCols)

  #aggDt_contDataOnly = modelPred_data[["userLevMeans_contDataOnly"]]
  #aggDt_withTreatData = modelPred_data[["userLevMeans_withTreatData"]]

  F = function(aggDt) {
    aggDt[ , "dummy"] = 1
    diffDf = DiffDf(
        df= aggDt,
        compareCol="expt_id_cfactual",
        itemCols="dummy",
        valueCols=valueCols,
        comparePair=comparePair,
        Diff=Diff,
        diffFcnList=NULL)

    vsStr = paste0(comparePair[1], "_vs_", comparePair[2])
    aggDiff = diffDf[ , mget(paste0(valueCols, "_", vsStr))]
  }

  return(list(
      "aggDiff_contDataOnly"=F(modelPred_data[["userLevMeans_contDataOnly"]]),
      "aggDiff_withTreatData"=F(modelPred_data[["userLevMeans_withTreatData"]]),
      "aggDiff_allDataNoExptId"=F(
          modelPred_data[["userLevMeans_allDataNoExptId"]])))
}

# check  imbalance in predictors in expt arms
Check_forImbalance = function(dt, predCols) {

  userCntDt = DtSimpleAgg(
      dt=dt,
      gbCols=c(predCols, "expt_id"),
      valueCols="user_id",
      F=function(x)length(unique(x)))

  userCntDf = data.frame(userCntDt)

  colName = paste0(predCols, collapse="_")
  userCntDf = Concat_stringColsDf(
      df=userCntDf,
      cols=predCols,
      colName="slice", sepStr="-")

  p = ggplot(
      userCntDf,
      aes(x=slice, y=user_id, fill=expt_id)) +
      geom_bar(stat="identity", width=.5, position="dodge") + ylab("") +
      xlab(colName) +
      guides(fill=guide_legend(title="user cnt")) +
      theme(
          text=element_text(size=16),
          axis.text.x=element_text(angle=30, hjust=1))

  return(list("p"=p, "userCntDt"=userCntDt))
}

## check adjustment balance
Check_adjustmentBalance = function(
    userDt_fromUsage_obs, predCols, valueCols,
    Diff, ss, savePlt=FALSE, fnSuffix="") {

  colSuffix = "_cont_factual_vs_cont_cfactual"
  valueCols2 = paste0(valueCols, colSuffix)

  infoDf = setNames(
        data.frame(matrix(ncol=3, nrow=0)),
        valueCols2)
  l = list(
      "aggDiff_contDataOnly"=infoDf,
      "aggDiff_withTreatData"=infoDf,
      "aggDiff_allDataNoExptId"=infoDf)

  methods = c(
      "aggDiff_contDataOnly",
      "aggDiff_withTreatData",
      "aggDiff_allDataNoExptId")

  methods2 = c(
      "control_data",
      "all_data",
      "all_no_expt_label")

  userNum = length(unique(userDt_fromUsage_obs[, user_id]))
  for (k in 1:1000) {
    userSample = sample(1:userNum, ss)
    userDt_fromUsage_obs2 = userDt_fromUsage_obs[user_id %in% userSample]

    res = PredBased_Diffs(
       userDt_fromUsage_obs=userDt_fromUsage_obs2,
       valueCols=valueCols,
       predCols=predCols,
       comparePair=c("cont_factual", "cont_cfactual"),
       Diff=Diff)

    for (method in methods) {
      l[[method]][nrow(l[[method]]) + 1, ] = res[[method]][1, get(valueCols2)]
    }
  }

  if (savePlt) {
    fn0 = paste0(figsPath, "checking_h_balance_", fnSuffix, ".png")
    print(fn0)
    fn = file(fn0, "w")
    Cairo(file=fn, type="png")
  }

  Plt = function() {
    F = log
    par(mfrow=c(3, 3))
    for (method in methods) {
      for (i in 1:3) {
        main =  mapvalues(method, from=methods, to=methods2)
        xlab = paste("log", valueCols2[i])
        xlab = gsub(x=xlab, "_cont_factual_vs_cont_cfactual", "")
        hist(F(l[[method]][ , valueCols2[i]]), col="blue", main=main,
             xlab=xlab, cex.lab=1.4, cex.axis=1.4, probability=TRUE)
      }
    }
  }

  Plt()

  if (savePlt) {
    dev.off()
    close(fn)
  }

  l[["Plt"]] = Plt

  return(l)
}

## Simple metrics calculation method which is based on diffs
# between treat and cont
# on the same valueCol. e.g.  "sum amount on treat" / "sum amount on expt"
# AggF
CalcDiffMetrics_userDt = function(
    userDt, compareCol, comparePair, valueCols, Diff, AggF) {

  ## then we aggregate with AggF which could be sum or mean
  aggDt = DtSimpleAgg(
      dt=userDt,
      gbCols=c(compareCol),
      valueCols=valueCols,
      F=AggF)

  userCntDt = DtSimpleAgg(
      dt=userDt,
      gbCols=c(compareCol),
      valueCols="user_id",
      F=function(x)length(unique(x)))

  names(userCntDt) = c(compareCol, "unique_usr_cnt")
  aggDt = merge(aggDt, userCntDt, by=c(compareCol))

  for (col in valueCols) {
    aggDt[ , paste0(col, "_per_user")] = (
        aggDt[ , get(col)] / aggDt[ , get("unique_usr_cnt")])
  }

  aggDt[ , "dummy"] = 1

  valueCols2 = c(valueCols, "unique_usr_cnt", paste0(valueCols, "_per_user"))

  ## calculate a difference data frame
  diffDf = DiffDf(
      df=aggDt,
      compareCol=compareCol,
      itemCols="dummy",
      valueCols=valueCols2,
      comparePair=comparePair,
      Diff=Diff,
      diffFcnList=NULL)

  vsStr = paste0(comparePair[1], "_vs_", comparePair[2])
  aggDiff = diffDf[ , mget(paste0(valueCols2, "_", vsStr))]
  return(aggDiff)
}

### Below we define some adjusted metrics using model predictions
## this adj is based on predictions on both arms assuming they are from control
# arm, the ratio version  is  AVG_{u in c} h(u, c) / AVG_{u in t} h(u, c)
# pred_cont_fac_mean: the prediction mean for control factual data
# pred_cont_cf_mean: the prediction on the control counterfactual data:
# this is the prediction for the treatment assuming they were on the control arm.
Metric_meanRatio = function(
    obs_treat_sum,
    obs_cont_sum,
    pred_cont_fac_mean=NULL,
    pred_cont_cf_mean=NULL,
    pred_treat_fac_mean=NULL,
    pred_treat_cf_mean=NULL,
    n_t=NULL,
    n_c=NULL,
    method="default",
    theta1=1/4) {

  obs_treat_mean = obs_treat_sum / n_t
  obs_cont_mean = obs_cont_sum / n_c

  if (method == "default") {
    return(obs_treat_mean / obs_cont_mean)
  }

  a = (pred_cont_fac_mean / pred_cont_cf_mean)^(theta1)
  return((obs_treat_mean / obs_cont_mean) * a)
}


Metric_ratioOfMeanRatios = function(
    obs_treat_sum1,
    obs_cont_sum1,
    obs_treat_sum2,
    obs_cont_sum2,
    pred_cont_fac_mean1=NULL,
    pred_cont_fac_mean2=NULL,
    pred_cont_cf_mean1=NULL,
    pred_cont_cf_mean2=NULL,
    pred_treat_fac_mean1=NULL,
    pred_treat_fac_mean2=NULL,
    pred_treat_cf_mean1=NULL,
    pred_treat_cf_mean2=NULL,
    n_t=NULL,
    n_c=NULL,
    method="default",
    theta1=1/2,
    theta2=1/2) {

  obs_treat_mean_numer = obs_treat_sum1 / n_t
  obs_cont_mean_numer = obs_cont_sum1 / n_c
  obs_treat_mean_denom = obs_treat_sum2 / n_t
  obs_cont_mean_denom = obs_cont_sum2 / n_c


  numer = obs_treat_mean_numer / obs_cont_mean_numer
  denom = obs_treat_mean_denom / obs_cont_mean_denom
  if (denom == 0) {
    warning("denom was zero")
    return(NULL)
  }

  if (method == "default") {
    return(numer / denom)
  }

  numer_adj = (pred_cont_fac_mean1 / pred_cont_cf_mean1)^(theta1)
  denom_adj = (pred_cont_fac_mean2 / pred_cont_cf_mean2)^(theta2)

  return((numer / denom) * (numer_adj / denom_adj))
}


Metric_sumRatio = function(
    obs_treat_sum,
    obs_cont_sum,
    pred_cont_fac_mean=NULL,
    pred_cont_cf_mean=NULL,
    pred_treat_fac_mean=NULL,
    pred_treat_cf_mean=NULL,
    n_t=NULL,
    n_c=NULL,
    method="default",
    theta1=1/4) {

  if (method == "default") {
    return(obs_treat_sum / obs_cont_sum)
  }

  a = (pred_cont_fac_mean / pred_cont_cf_mean)^(theta1)
  return((obs_treat_sum / obs_cont_sum) * a)
}

## this is for mean diff: "-"
Metric_meanMinus = function(
    obs_treat_sum,
    obs_cont_sum,
    pred_cont_fac_mean=NULL,
    pred_cont_cf_mean=NULL,
    pred_treat_fac_mean=NULL,
    pred_treat_cf_mean=NULL,
    n_t=NULL,
    n_c=NULL,
    method="default",
    theta1=1/4) {

  if (method == "default") {
    return(obs_treat_sum/n_t - obs_cont_sum/n_c)
  }

  return(
      obs_treat_sum/n_t - obs_cont_sum/n_c +
      theta1 * (pred_cont_fac_mean - pred_cont_cf_mean))
}

## this is an adjustment for metrics which are sum of usage differences
# therefore we need to use n_t and n_c in our adj
Metric_sumMinus = function(
    obs_treat_sum,
    obs_cont_sum,
    pred_cont_fac_mean=NULL,
    pred_cont_cf_mean=NULL,
    pred_treat_fac_mean=NULL,
    pred_treat_cf_mean=NULL,
    n_t=NULL,
    n_c=NULL,
    method="default",
    theta1=1/2) {

  if (method == "default") {
    return(obs_treat_sum - obs_cont_sum)
  }

  return(
      obs_treat_sum - obs_cont_sum -
      n_c * theta1 * (pred_cont_cf_mean - pred_cont_fac_mean))
}

## this is an adjustment for metrics which are sum of usage differences
# therefore we need to use n_t and n_c in our adj
Metric_sumMinus2 = function(
    obs_treat_sum,
    obs_cont_sum,
    pred_cont_fac_mean=NULL,
    pred_cont_cf_mean=NULL,
    pred_treat_fac_mean=NULL,
    pred_treat_cf_mean=NULL,
    n_t=NULL,
    n_c=NULL,
    method="default",
    theta1=1/2) {

  if (method == "default") {
    return(obs_treat_sum - obs_cont_sum)
  }

  a = theta1 * (pred_cont_fac_mean / pred_cont_cf_mean)

  return(obs_treat_sum * a - obs_cont_sum)
}

## takes observed and model predicted data
# and an Adjusted metric (Adj) to calculate adj metrics
# $adjDiff_withTreatData
# imp_count_treat_vs_cont obs_interact_treat_vs_cont obs_amount_treat_vs_cont
# bivarMetric = list(F=F, col1, col2)

CalcAdjMetrics_aggData = function(
    obsSumAggDt,
    modelPred_data,
    valueCols,
    n_t,
    n_c,
    CommonMetric=NULL,
    metricList=NULL,
    bivarMetric=NULL) {


  AdjF_univar = function(modStr) {

    userLevMeans = modelPred_data[[paste0("userLevMeans_", modStr)]]
    infoDf = modelPred_data[[paste0("infoDf_", modStr)]]

    valueCols2 = paste0(valueCols, "_treat_vs_cont")
    metricDf = setNames(
        data.frame(matrix(ncol=length(valueCols), nrow=0)),
        valueCols2)

    x = rep(NA, length(valueCols))
    for (i in 1:length(valueCols)) {

      col = valueCols[i]
      theta1 = infoDf[infoDf["valueCol"] == col, "theta"]
      obs_treat_sum = obsSumAggDt[expt_id == "treat" , get(col)]
      obs_cont_sum = obsSumAggDt[expt_id == "cont" , get(col)]
      pred_cont_fac_mean = userLevMeans[expt_id_cfactual == "cont_factual", get(col)]
      pred_cont_cf_mean = userLevMeans[expt_id_cfactual == "cont_cfactual", get(col)]
      pred_treat_fac_mean = userLevMeans[expt_id_cfactual == "treat_factual", get(col)]
      pred_treat_cf_mean = userLevMeans[expt_id_cfactual == "treat_cfactual", get(col)]

      Metric = CommonMetric
      if (!is.null(metricList)) {
        Metric = metricList[[i]]
      }

      x[i] = Metric(
          obs_treat_sum=obs_treat_sum,
          obs_cont_sum=obs_cont_sum,
          pred_cont_fac_mean=pred_cont_fac_mean,
          pred_cont_cf_mean=pred_cont_cf_mean,
          pred_treat_fac_mean=pred_treat_fac_mean,
          pred_treat_cf_mean=pred_treat_cf_mean,
          n_t=n_t,
          n_c=n_c,
          method="adjusted",
          theta1=theta1)
    }

    metricDf[1, ] = x
    return(metricDf)
  }

  RawF_univar = function() {

    valueCols2 = paste0(valueCols, "_treat_vs_cont")
    metricDf = setNames(
        data.frame(matrix(ncol=length(valueCols), nrow=0)),
        valueCols2)

    x = rep(NA, length(valueCols))
    thetaVec = rep(NA, length(valueCols))

    for (i in 1:length(valueCols)) {

      col = valueCols[i]
      obs_treat_sum = obsSumAggDt[expt_id == "treat" , get(col)]
      obs_cont_sum = obsSumAggDt[expt_id == "cont" , get(col)]

      Metric = CommonMetric
      if (!is.null(metricList)) {
        Metric = metricList[[i]]
      }

      x[i] = Metric(
          obs_treat_sum=obs_treat_sum,
          obs_cont_sum=obs_cont_sum,
          pred_cont_fac_mean=NULL,
          pred_cont_cf_mean=NULL,
          pred_treat_fac_mean=NULL,
          pred_treat_cf_mean=NULL,
          n_t=n_t,
          n_c=n_c,
          method="default",
          theta1=NULL)
    }

    metricDf[1, ] = x

    return(metricDf)
  }

  AdjF_bivar = function(modStr) {

    userLevMeans = modelPred_data[[paste0("userLevMeans_", modStr)]]
    infoDf = modelPred_data[[paste0("infoDf_", modStr)]]

    col1 = bivarMetric[["col1"]]
    col2 = bivarMetric[["col2"]]
    metricColName = paste0(col1, "_over_", col2, "_treat_vs_cont")


    metricDf = setNames(
        data.frame(matrix(ncol=1, nrow=0)),
        metricColName)

    theta1 = infoDf[infoDf["valueCol"] == col1, "theta"]
    theta2 = infoDf[infoDf["valueCol"] == col2, "theta"]

    obs_treat_sum1 = obsSumAggDt[expt_id == "treat" , get(col1)]
    obs_cont_sum1 = obsSumAggDt[expt_id == "cont" , get(col1)]
    pred_cont_fac_mean1 = userLevMeans[expt_id_cfactual == "cont_factual", get(col1)]
    pred_cont_cf_mean1 = userLevMeans[expt_id_cfactual == "cont_cfactual", get(col1)]
    pred_treat_fac_mean1 = userLevMeans[expt_id_cfactual == "treat_factual", get(col1)]
    pred_treat_cf_mean1 = userLevMeans[expt_id_cfactual == "treat_cfactual", get(col1)]

    obs_treat_sum2 = obsSumAggDt[expt_id == "treat" , get(col2)]
    obs_cont_sum2 = obsSumAggDt[expt_id == "cont" , get(col2)]
    pred_cont_fac_mean2 = userLevMeans[expt_id_cfactual == "cont_factual", get(col2)]
    pred_cont_cf_mean2 = userLevMeans[expt_id_cfactual == "cont_cfactual", get(col2)]
    pred_treat_fac_mean2 = userLevMeans[expt_id_cfactual == "treat_factual", get(col2)]
    pred_treat_cf_mean2 = userLevMeans[expt_id_cfactual == "treat_cfactual", get(col2)]


    Metric = bivarMetric[["F"]]

    x = Metric(
      obs_treat_sum1=obs_treat_sum1,
      obs_cont_sum1=obs_cont_sum1,
      obs_treat_sum2=obs_treat_sum2,
      obs_cont_sum2=obs_cont_sum2,
      pred_cont_fac_mean1=pred_cont_fac_mean1,
      pred_cont_fac_mean2=pred_cont_fac_mean2,
      pred_cont_cf_mean1=pred_cont_cf_mean1,
      pred_cont_cf_mean2=pred_cont_cf_mean2,
      pred_treat_fac_mean1=pred_treat_fac_mean1,
      pred_treat_fac_mean2=pred_treat_fac_mean2,
      pred_treat_cf_mean1=pred_treat_cf_mean1,
      pred_treat_cf_mean2=pred_treat_cf_mean2,
      n_t=n_t,
      n_c=n_c,
      method="adjusted",
      theta1=theta1,
      theta2=theta2)


    metricDf[1, ] = x
    return(metricDf)
  }

  RawF_bivar = function(modStr) {

    col1 = bivarMetric[["col1"]]
    col2 = bivarMetric[["col2"]]
    metricColName = paste0(col1, "_over_", col2, "_treat_vs_cont")

    metricDf = setNames(
        data.frame(matrix(ncol=1, nrow=0)),
        metricColName)

    obs_treat_sum1 = obsSumAggDt[expt_id == "treat" , get(col1)]
    obs_cont_sum1 = obsSumAggDt[expt_id == "cont" , get(col1)]

    obs_treat_sum2 = obsSumAggDt[expt_id == "treat" , get(col2)]
    obs_cont_sum2 = obsSumAggDt[expt_id == "cont" , get(col2)]


    Metric = bivarMetric[["F"]]

    x = Metric(
      obs_treat_sum1=obs_treat_sum1,
      obs_cont_sum1=obs_cont_sum1,
      obs_treat_sum2=obs_treat_sum2,
      obs_cont_sum2=obs_cont_sum2,
      pred_cont_fac_mean1=NULL,
      pred_cont_fac_mean2=NULL,
      pred_cont_cf_mean1=NULL,
      pred_cont_cf_mean2=NULL,
      pred_treat_fac_mean1=NULL,
      pred_treat_fac_mean2=NULL,
      pred_treat_cf_mean1=NULL,
      pred_treat_cf_mean2=NULL,
      n_t=n_t,
      n_c=n_c,
      method="default",
      theta1=NULL,
      theta2=NULL)

    metricDf[1, ] = x
    return(metricDf)
  }


  RawFcn = function() {
    if (is.null(bivarMetric)) {
      return(RawF_univar)
    }
    return(RawF_bivar)
  }

  AdjFcn = function() {
    if (is.null(bivarMetric)) {
      return(AdjF_univar)
    }
    return(AdjF_bivar)
  }

  adjDiffList = list(
        "raw"=RawFcn()(),
        "adjDiff_contDataOnly"=AdjFcn()("contDataOnly"),
        "adjDiff_withTreatData"=AdjFcn()("withTreatData"),
        "adjDiff_allDataNoExptId"=AdjFcn()("allDataNoExptId"))
}

## this is for adj version
# it takes obs userDt as input
# it fits models, returns adjusted metrics
## output format:
# $adjDiff_withTreatData
# imp_count_treat_vs_cont obs_interact_treat_vs_cont obs_amount_treat_vs_cont
CalcAdjMetrics_fromUserDt = function(
    userDt_fromUsage_obs,
    predCols,
    valueCols,
    CommonMetric=NULL,
    metricList=NULL,
    bivarMetric=NULL) {

  userCntDt = DtSimpleAgg(
      dt=userDt_fromUsage_obs,
      gbCols="expt_id",
      valueCols="user_id",
      F=function(x)length(unique(x)))

  n_t = userCntDt[expt_id == "treat", user_id]
  n_c = userCntDt[expt_id == "cont", user_id]

  ## obs sum aggregates
  obsSumAggDt = DtSimpleAgg(
      dt=userDt_fromUsage_obs,
      gbCols=c("expt_id"),
      valueCols=valueCols,
      F=sum)

  ## model mean aggregates
  #TODO: Reza Hosseini: we could replace that with all data for testing
  modelPred_data = PredBased_userLevelMeans(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      valueCols=valueCols,
      predCols=predCols)

  adjMetricList = CalcAdjMetrics_aggData(
      obsSumAggDt=obsSumAggDt,
      modelPred_data=modelPred_data,
      valueCols=valueCols,
      n_t=n_t,
      n_c=n_c,
      CommonMetric=CommonMetric,
      metricList=metricList,
      bivarMetric=bivarMetric)

  return(adjMetricList)
}

TestCalcAdjMetrics_fromUserDt = function() {

  simData = SimUsage_checkResults(
      userNum=500, predCols=c("country", "gender"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"))

  userDt_fromUsage_obs = simData[["userDt_fromUsage_obs"]]

  ## example 1
  metricInfo = list(
      "name"="mean_ratio", "Metric"=Metric_meanRatio, "AggF"=mean,
      "Diff"=function(x, y) {x / y})

  adjMetrics = CalcAdjMetrics_fromUserDt(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      predCols=c("country", "gender"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      CommonMetric=metricInfo[["Metric"]],
      metricList=NULL)

  rawMetrics = CalcDiffMetrics_userDt(
      userDt=userDt_fromUsage_obs,
      compareCol="expt_id",
      comparePair=c("treat", "cont"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      Diff=metricInfo[["Diff"]],
      AggF=metricInfo[["AggF"]])

  Mark(adjMetrics, "adjMetrics")
  Mark(rawMetrics, "rawMetrics")

  ## example 2
  metricInfo = list(
      "name"="sum_ratio", "Metric"=Metric_sumRatio, "AggF"=sum,
      "Diff"=function(x, y) {x / y})

  adjMetrics = CalcAdjMetrics_fromUserDt(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      predCols=c("country", "gender"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      CommonMetric=metricInfo[["Metric"]],
      metricList=NULL)

  rawMetrics = CalcDiffMetrics_userDt(
      userDt=userDt_fromUsage_obs,
      compareCol="expt_id",
      comparePair=c("treat", "cont"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      Diff=metricInfo[["Diff"]],
      AggF=metricInfo[["AggF"]])

  Mark(adjMetrics, "adjMetrics")
  Mark(rawMetrics, "rawMetrics")

  ## example 3: bivar metrics
  bivarMetric = list(
      "F"=Metric_ratioOfMeanRatios,
      "col1"="obs_interact",
      "col2"="imp_count")

  adjMetrics_bivar = CalcAdjMetrics_fromUserDt(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      predCols=c("country", "gender"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      CommonMetric=NULL,
      metricList=NULL,
      bivarMetric=bivarMetric)

  # compare to the num and denom metrics to see if ratio matches
  metricInfo = list(
      "name"="mean_ratio", "Metric"=Metric_meanRatio, "AggF"=mean,
      "Diff"=function(x, y) {x / y})

  adjMetrics_univar = CalcAdjMetrics_fromUserDt(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      predCols=c("country", "gender"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      CommonMetric=metricInfo[["Metric"]],
      metricList=NULL)

  col1 =  bivarMetric[["col1"]]
  col2 =  bivarMetric[["col2"]]
  newCol = paste0(col1, "_over_", col2, "_treat_vs_cont")
  col1_aug = paste0(col1, "_treat_vs_cont")
  col2_aug = paste0(col2, "_treat_vs_cont")

  for (method in names(adjMetrics_univar)) {
      adjMetrics_univar[[method]][ , newCol] = (
          adjMetrics_univar[[method]][ , col1_aug] /
          adjMetrics_univar[[method]][ , col2_aug]
        )
  }

  adjMetrics_bivar
  adjMetrics_univar
}

# generates a function which applies Metric on a df
# for each element of comparePair = (cont, treat)
# and returns the Diff between the two
ExptMetricFcn = function(Metric, Diff, compareCol, comparePair) {

  F = function(dt) {
    dtList = list()
    metricValues = c(
        Metric(dt[get(compareCol) == comparePair[1]]),
        Metric(dt[get(compareCol) == comparePair[2]]))

    return(Diff(metricValues[1], metricValues[2]))
  }

  return(F)
}

## this calculates a given metrics CIs for the default method
# and adjusted methods
# two methods are implemented: simple_bucket, jackknife
# parallelization is provided as an option
CalcMetricCis_withBuckets = function(
    dt, valueCols, predCols, CommonMetric=NULL, metricList=NULL,
    bivarMetric=NULL, ci_method="simple_bucket",
    parallel=FALSE, maxCoreNum=NULL, parallel_outfile="") {

  methods = c(
      "raw", "adjDiff_contDataOnly", "adjDiff_withTreatData",
      "adjDiff_allDataNoExptId")

  CalcMetrics = function(x) {
    #Src()

    if (ci_method == "simple_bucket") {
      dt2 = dt[bucket == x]
    } else {
      dt2 = dt[bucket != x]
    }

    adjMetrics = CalcAdjMetrics_fromUserDt(
        userDt_fromUsage_obs=dt2,
        predCols=predCols,
        valueCols=valueCols,
        CommonMetric=CommonMetric,
        metricList=metricList,
        bivarMetric= bivarMetric)

    return(adjMetrics)
  }

  buckets = unique(dt[ , bucket])
  b = length(buckets)
  ## t-distribution value for CI construction
  thresh = qt(
      p=1-0.025, df=b, ncp=0, lower.tail=TRUE, log.p=FALSE)

  options(warn=-1)

  if (!parallel) {
    estimList = lapply(X=as.list(buckets), FUN=CalcMetrics)
  } else {
    suppressMessages(library("parallel"))
    closeAllConnections()
    no_cores = detectCores() - 3
    no_cores = min(no_cores, b + 1, maxCoreNum)
    Mark(no_cores, "no_cores")
    # Initiate cluster
    cl = makeCluster(no_cores, outfile=parallel_outfile)
    clusterExport(
            cl=cl,
            list(
                "dt", "valueCols", "predCols", "CommonMetric", "metricList",
                "bivarMetric", "ci_method", "Src"),
            envir=environment())
    estimList =  parLapply(cl=cl, X=as.list(buckets), fun=CalcMetrics)
    stopCluster(cl)
    closeAllConnections()
  }

  estimDfList = list()

  ## combine different methods metrics into one dataframe for each method
  for (method in methods) {
    estimDfList[[method]] = do.call(
        what=rbind,
        args=lapply(
            X=1:length(estimList),
            FUN=function(i){estimList[[i]][[method]]}))
  }

  if (ci_method == "simple_bucket") {
    SdMultiplier = function(n) {sqrt(1 / n)}
  } else {
    SdMultiplier = function(n) {sqrt(n-1)}
  }

  CalcCi = function(method) {
    df = estimDfList[[method]]
    outDf = setNames(
        data.frame(matrix(ncol=5, nrow=0)),
        c("mean", "sd", "ci_lower", "ci_upper", "ci_length"))

    for (i in 1:length(names(df))) {
      col = names(df)[i]
      m = mean(df[ , col])
      s = sd(df[ , col])
      r = thresh * s * SdMultiplier(nrow(df))
      ci_lower = m - r
      ci_upper = m + r
      ci_length = ci_upper - ci_lower
      outDf[i, ] = c(m, s, ci_lower, ci_upper, ci_length)
    }

    outDf[ , "resp"] = names(df)
    outDf[ , "method"] = method
    outDf[ , "ci_method"] = ci_method
    return(outDf)
  }

  ciDf = do.call(what=rbind, args=lapply(X=methods, FUN=CalcCi))
  ciDf = ciDf[order(ciDf[ , "resp"]), ]
  return(list(
      "ciDf"=ciDf, "estimList"=estimList, "estimDfList"=estimDfList))
}

## calculating CI with bootstrap
CalcMetricCis_withBootstrap = function(
    dt, valueCols, predCols, CommonMetric=NULL, metricList=NULL,
    bivarMetric=NULL, bsNum=300, parallel=FALSE, parallel_outfile="") {

  methods = c(
      "raw", "adjDiff_contDataOnly", "adjDiff_withTreatData",
      "adjDiff_allDataNoExptId")

  n = nrow(dt)
  dt2 = copy(dt)

  ## calculates metrics for a bootstrapped sample of data
  # this returns raw metrics as well as adjusted
  CalcMetrics = function(x) {

    Src()
    samp = sample(1:n, n, replace=TRUE)
    dt2 = dt2[samp, ]
    dt2[ , "user_id"] = 1:n

    adjMetrics = CalcAdjMetrics_fromUserDt(
        userDt_fromUsage_obs=dt2,
        predCols=predCols,
        valueCols=valueCols,
        CommonMetric=CommonMetric,
        metricList=metricList,
        bivarMetric=bivarMetric)

    return(adjMetrics)
  }

  options(warn=-1)

  if (!parallel) {
    estimList = lapply(X=1:bsNum, FUN=CalcMetrics)
  }  else {
    suppressMessages(library("parallel"))
    closeAllConnections()
    no_cores = detectCores() - 3
    no_cores = min(no_cores, length(bsNum) + 1)
    Mark(no_cores, "no_cores")
    # Initiate cluster
    cl = makeCluster(no_cores, outfile=parallel_outfile)
    clusterExport(
            cl=cl,
            list(
                "dt", "valueCols", "predCols", "CommonMetric", "metricList",
                "bivarMetric", "Src"),
            envir=environment())
    estimList =  parLapply(cl=cl, X=1:bsNum, fun=CalcMetrics)
    stopCluster(cl)
    closeAllConnections()
  }

  ## this is a list of data frames per method
  estimDfList = list()
  ## combine different methods metrics into one data frame for each method
  for (method in methods) {
    estimDfList[[method]] = do.call(
        what=rbind,
        args=lapply(
            X=1:length(estimList),
            FUN=function(i){estimList[[i]][[method]]}))
  }

  CalcCi = function(method) {
    df = estimDfList[[method]]
    outDf = setNames(
        data.frame(matrix(ncol=5, nrow=0)),
        c("mean", "sd", "ci_lower", "ci_upper", "ci_length"))

    for (i in 1:length(names(df))) {
      col = names(df)[i]
      m = mean(df[ , col])
      s = sd(df[ , col])
      ci_lower = quantile(df[ , col], 0.025)
      ci_upper = quantile(df[ , col], 0.975)
      ci_length = ci_upper - ci_lower
      outDf[i, ] = c(m, s, ci_lower, ci_upper, ci_length)
    }

    outDf[ , "resp"] = names(df)
    outDf[ , "method"] = method
    outDf[ , "ci_method"] = "bootstrap"
    return(outDf)
  }

  ciDf = do.call(what=rbind, args=lapply(X=methods, FUN=CalcCi))
  ciDf = ciDf[order(ciDf[ , "resp"]), ]
  rownames(ciDf) = NULL
  return(list(
      "ciDf"=ciDf, "estimList"=estimList, "estimDfList"=estimDfList))
}

## comparing the standard deviations coming from various models
CompareMethodsSd = function(
    estimDt, methods, valueCols, mainSuffix="", sizeAlpha=1.5) {

  sdDt = DtSimpleAgg(dt=estimDt, gbCols=c("ss", "method"), F=sd)
  sdDt = sdDt[method %in% methods]
  sdDt = DtSubsetCols(sdDt, dropCols="sim_num")

  wideDt = dcast(
      sdDt, ss ~ method,
      value.var=valueCols)
  oldMethod = methods[1]
  newMethods = methods[-1]

  ratioColsList = list()
  for (col in valueCols) {
    ratioCols = NULL
    for (method in newMethods) {

      ratioCol = paste0(col, "_", method, "_vs_", oldMethod)
      ratioCols = c(ratioCols, ratioCol)
      wideDt[ , ratioCol] = (
          wideDt[ , get(paste0(col, "_", method))] /
          wideDt[ , get(paste0(col, "_", oldMethod))])
    }
    ratioColsList[[col]] = ratioCols
  }

  df = data.frame(wideDt)
  pltFcnList = list()

  for (col in valueCols) {
    yCols = ratioColsList[[col]]
    xCol = "ss"
    ylab = "ratio"
    pltFcnList[[col]] = local({
        main = paste(col, mainSuffix, sep="; ")
        yCols2 = yCols
        function() {
          PltDfColsLines(
              df=df, xCols="ss", ylim=c(0, 1.5), yCols=yCols2,
              ylab="ratio", xlab="user_num",
              main=main, varLty=TRUE, sizeAlpha=1.5)}})
  }

  Plt = function() {
    par(mfrow=c(ceiling(length(valueCols) / 2), 2))
    for (col in valueCols) {
      pltFcnList[[col]]()
    }
  }

  return(list(
      "wideDt"=wideDt,
      "ratioColsList"=ratioColsList,
      "pltFcnList"=pltFcnList,
      "Plt"=Plt))
}

## calc convg of Ci with increasing sample size
CiLengthConvg = function(
    dt, gridNum, valueCols, predCols, metricList=NULL,
    CommonMetric=NULL, bivarMetric=NULL, bucketNum=50,
    bs=FALSE, bsNum=300,
    compareValues=c("raw", "control_data", "all_data"),
    userNumProp=NULL, parallel=FALSE, parallel_outfile="",
    maxCoreNum=10, mainSuffix="", minSs=1000) {

  Mod = GenModFcn(bucketNum)
  dt[ , "bucket"] = Mod(as.numeric(dt[ , user_id]))

  userNum = nrow(dt)
  if (nrow(dt) != length(unique(dt[ , user_id]))) {
    warning("data table is not a user data table")
    warning("each row does not correspond with a unique user.")
    dt = dt[!duplicated(dt[ , user_id]), ]
    #return(NULL)
  }

  users = unique(dt[ , user_id])
  userNum = length(users)

  Samp = function(n) {
    set.seed(n)
    userSample = sample(users, n)
    dt2 = dt[user_id %in% userSample]
    return(dt2)
  }

  Jk = function(n) {
    Src()
    dt2 = Samp(n)
    res = CalcMetricCis_withBuckets(
        dt=dt2, valueCols=valueCols, predCols=predCols,
        CommonMetric=CommonMetric, metricList=metricList,
        bivarMetric=bivarMetric,
        ci_method="jk_bucket",
        parallel=FALSE)
    ciDf_jk = res[["ciDf"]]
    ciDf_jk[ , "ss"] = n
    return(ciDf_jk)
  }

  Bs = function(n) {
    Src()
    dt2 = Samp(n)
    res = CalcMetricCis_withBootstrap(
        dt=dt2, valueCols=valueCols, predCols=predCols,
        CommonMetric=CommonMetric, metricList=metricList,
        bivarMetric=bivarMetric,
        bsNum=bsNum, parallel=FALSE)
    ciDf_bs = res[["ciDf"]]
    ciDf_bs[ , "ss"] = n
    return(ciDf_bs)
  }

  if (!is.null(userNumProp)) {
    userNum = (userNum * userNumProp)
  }

  step = round(userNum / gridNum)
  print(step)
  init = max(c(step, minSs))
  print(init)
  x =  seq(init, userNum, by=step)

  if (!parallel) {
    jkDfList = lapply(FUN=Jk, X=x)
  } else {
      suppressMessages(library("parallel"))
      closeAllConnections()
      no_cores = detectCores() - 2
      no_cores = min(no_cores, length(x) + 1, maxCoreNum)
      Mark(no_cores, "no_cores")
      # Initiate cluster
      cl = makeCluster(no_cores, outfile=parallel_outfile)
      clusterExport(
          cl=cl,
          list(
            "dt", "valueCols", "predCols", "CommonMetric",
            "metricList", "bivarMetric", "Src", "Jk",
            "users", "Samp"),
            envir=environment())
    jkDfList = parLapply(cl=cl, X=x, fun=Jk)
    stopCluster(cl)
    closeAllConnections()
  }

  jkDf = do.call(what=rbind, args=jkDfList)

  bsDf = NULL
  if (bs) {
    if (!parallel) {
      bsDfList = lapply(FUN=Bs, X=x)
    } else {
      suppressMessages(library("parallel"))
      closeAllConnections()
      no_cores = detectCores() - 2
      no_cores = min(no_cores, length(x) + 1)
      Mark(no_cores, "no_cores")
      # Initiate cluster
      cl = makeCluster(no_cores, outfile=parallel_outfile)
      clusterExport(
          cl=cl,
          list(
            "dt", "valueCols", "predCols", "CommonMetric", "metricList",
            "bivarMetric", "Src", "bsNum", "Bs", "users"),
            envir=environment())
    bsDfList = parLapply(cl=cl, X=x, fun=Bs)
    bsDf = do.call(what=rbind, args=bsDfList)
    }
  }

  Plt = function(df, metrics=NULL, values=compareValues) {
    #metrics = paste0(valueCols, "_treat_vs_cont")
    df[ , "method"] = mapvalues(
        df[ , "method"],
        from=c("raw", "adjDiff_contDataOnly", "adjDiff_withTreatData"),
        to=c("raw", "control_data", "all_data"))
    if (is.null(metrics)) {
      metrics = unique(df[ , "resp"])
    }

    rowNum = ceiling(sqrt(length(metrics)))
    colNum = rowNum
    par(mfrow=c(rowNum, colNum))

    for (metric in metrics) {
      df2 = df[df[ , "resp"] == metric, ]
      df2[ , "user_num"] = df2[ , "ss"]
      Plt_compareCiGroups(
          df=df2, xCol="user_num", lowerCol="ci_lower", upperCol="ci_upper",
          compareCol="method", compareValues=values,
          ylab="", main=paste(metric, mainSuffix, sep=""))
    }
  }

  return(list("jkDf"=jkDf, "bsDf"=bsDf, "Plt"=Plt))
}

## this calculates the sample size gain by doing var reduction
VarReduct_sampleSizeGain = function(r) {
  return(1 / (1 - r^2))
}

Plt_adjCiSampleSizeGain = function(
    figsPath="", main="CI length reduction by adjustment") {


  Plt = function() {

    x1 = (1:950) / 1000
    y1 = VarReduct_sampleSizeGain(x1)

    x2 = (1:9) / 10
    y2 = VarReduct_sampleSizeGain(x2)


    plot(
        x1, y1,
        xlab="cross-validated corr of model and observed",
        ylab="sample size multiplier",
        lwd=2, col=ColAlpha("blue", 0.6), type='l',
        cex.lab=1.2, cex.main=1.2, cex.axis=1.1, main=main, log="y")
    points(x2, y2, pch=20, col=ColAlpha("blue", 0.6))
    grid(lwd=1.5, col=ColAlpha("red", 0.5), ny=10)
  }

  Plt()

  fn0 = paste0(figsPath, "var_reduct_sample_size_gain.png")
  fn = file(fn0, "w")
  Mark(fn0, "filename")

  r = 1.2
  Cairo(
      width=640*r, height=480*r, file=fn, type="png", dpi=110*r,
      pointsize=20*r)

  Plt()
  dev.off()
  close(fn)
}

### plot expected reduction as a function of correlation by adjustment
Plt_adjCiLengthReduct = function(
    figsPath="", main="CI length reduction by adjustment") {

  Plt = function() {

    Reduct = function(r) {
      1 - sqrt((1-r^2))
    }

    x1 = (0:100) / 100
    y1 = Reduct(x1)

    x2 = (0:10) / 10
    y2 = Reduct(x2)
    plot(
        x1, y1,
        xlab="cv corr of model and observed",
        ylab="CI length reduction",
        lwd=2, col=ColAlpha("blue", 0.6), type='l',
        cex.lab=1.2, cex.main=1.2, cex.axis=1.1, main=main)
    points(x2, y2, pch=20, col=ColAlpha("blue", 0.6))
    grid(lwd=1.5, col=ColAlpha("red", 0.5))
    abline(
        h=seq(1, 10, 2)/10, v=seq(1, 10, 2)/10,
        col=ColAlpha("grey", 0.8), lty=3, lwd=1.5)
  }

  fn0 = paste0(figsPath, "ci_length_reduction.png")
  fn = file(fn0, "w")
  Mark(fn0, "filename")

  r = 1.2
  Cairo(
      width=640*r, height=480*r, file=fn, type="png", dpi=110*r,
      pointsize=20*r)

  Plt()
  dev.off()
  close(fn)
  return(Plt)
}

### plot expected reduction as a function of sample size by adjustment
Plt_ssCiLengthReduct = function(
    figsPath="", main="CI length reduction by sample size inc.") {

  Plt = function() {

    Reduct = function(k) {
      1 - sqrt(1/k)
    }

    x1 = (100:1000)/100
    y1 = Reduct(x1)

    x2 = (1:10)
    y2 = Reduct(x2)

    plot(
        x1, y1,
        xlab="increase in sample size",
        ylab="CI length reduction",
        lwd=2, col=ColAlpha("blue", 0.6), type='l',
        cex.lab=1.2, cex.main=1.2, cex.axis=1.1,
        xaxt="n", main=main)
    axis(1, at=x2, labels=x2)
    points(x2, y2, pch=20, col=ColAlpha("blue", 0.6))
    grid(lwd=1.5, col=ColAlpha("red", 0.5))
    abline(
        h=seq(1, 10, 2)/10, v=x2,
        col=ColAlpha("grey", 0.8), lty=3, lwd=1.5)
  }

  fn0 = paste0(figsPath, "ci_length_reduction_ss.png")
  fn = file(fn0, "w")
  Mark(fn0, "filename")

  r = 1.2
  Cairo(
        width=640*r, height=480*r, file=fn, type="png", dpi=110*r,
        pointsize=20*r)

  Plt()
  dev.off()
  close(fn)
  return(Plt)
}

## birth year to age
BirthYear_toAgeCateg = function(x, currentYear) {
  if (is.na(x) | is.null(x) | x == "" | x == 0) {
    return("other")
  }

  x = as.numeric(x)
  age = currentYear - x

  if (age <= 17) {
    return("<18")
  }

  if (age <= 25) {
    return("18-25")
  }

  if (age <= 35) {
    return("26-35")
  }

  if (age <= 50) {
    return("36-50")
  }

  return(">51")
}



## calculates pre-post metrics
Calc_prePostMetrics = function(
    dt, valueCol, prePostCol, compareCol, comparePair,
    AggF=mean) {

  aggDt = dt[ , AggF(get(valueCol)), by=mget(c(compareCol, prePostCol))]

  colnames(aggDt) = c(compareCol, prePostCol, valueCol)

  metricsDt = dcast(
      aggDt,
      as.formula(paste0(compareCol, "~", prePostCol)),
      value.var=valueCol)

  metricsDt = metricsDt[ , mget(c(compareCol, "pre", "post"))]

  colnames(metricsDt) = c(compareCol, paste0(valueCol, c( "_pre", "_post")))

  prePostValueCol = paste0(valueCol, "_post_over_pre")

  metricsDt[ , prePostValueCol] = (
      metricsDt[ , get(paste0(valueCol, "_post"))] /
      metricsDt[ , get(paste0(valueCol, "_pre"))])

  prePostMetricsDt = metricsDt

  metricsDt[ , "dummy_var"] = "1"

  compareMetricsDt = dcast(
      metricsDt,
      as.formula(paste0("dummy_var ~", compareCol)),
      value.var = c(prePostValueCol, paste0(valueCol, c( "_pre", "_post"))))

  compareMetricsDt = DtSubsetCols(compareMetricsDt, dropCols="dummy_var")

  cols = setdiff(colnames(metricsDt), c("dummy_var", compareCol))
  compareStr = paste0(comparePair[2], "_over_", comparePair[1])


  for (col in cols) {

    col1 = paste0(col, "_", comparePair[1])
    col2 = paste0(col, "_", comparePair[2])

    newCol = paste0(col, "_", compareStr)

    compareMetricsDt[ , newCol] = (

      compareMetricsDt[ , get(col2)] / compareMetricsDt[ , get(col1)]

    )

  }

  return(list(
      "prePostDt"=prePostMetricsDt, "compareDt"=compareMetricsDt))

}


TestCalc_prePostMetrics = function() {

  k = 1000
  df = setNames(
    data.frame(matrix(ncol=4, nrow=k)),
    c("id", "pre_post", "expt_id", "value"))

  df[ , "id"] = 1:k
  df[ , "pre_post"] = c(rep("pre", k/2), rep("post", k/2))
  df[ , "expt_id"] = sample(c("treat", "cont"), k, replace=TRUE)
  df[ , "value"] = c(abs(rnorm(k/2)) + 5, abs(rnorm(k/2)))


  res = Calc_prePostMetrics(
      dt=data.table(df),
      valueCol="value",
      prePostCol="pre_post",
      compareCol="expt_id",
      comparePair=c("cont", "treat"),
      AggF=mean)

}
