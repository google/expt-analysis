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


# simulate usage data end to end
SimUsageDf_e2e = function(
    userNum,
    timestamp1="2018/06/01",
    timestamp2="2018/06/07",
    noTimestamps=FALSE,
    userDistbns=list(
        "country"=list("US"=0.4, "IN"=0.3, "BR"=0.3),
        "gender"=list("MALE"=0.5, "FEMALE"=0.5),
        "expt_id"=list("cont"=0.5, "treat"=0.5)),
    valueCols=c("count", "amount", "interact"),
    mus=c(0, 0, 0),
    valueCols_covars = list(
        "count"=list(
            "country"=list("US"=0, "IN"=1, "BR"=2),
            "gender"=list("MALE"=0, "FEMALE"=1),
            "expt_id"=list("cont"=0, "treat"=1)),
        "amount"=list(
            "country"=list("US"=0, "IN"=1, "BR"=2),
            "gender"=list("MALE"=0, "FEMALE"=1),
            "expt_id"=list("cont"=0, "treat"=1)),
        "interact"=list(
            "country"=list("US"=0, "IN"=1, "BR"=2),
            "gender"=list("MALE"=0, "FEMALE"=1),
            "expt_id"=list("cont"=0, "treat"=1))),
    userVarCov_chol=diag(3), #matrix(rnorm(9, mean=0, sd=0.1), 3, 3),
    transfList=list(
        "count"=function(x){1.5^(x)},
        "amount"=function(x){1.5^(x)},
        "interact"=function(x){LogitInv(x)}),
    valueDistbns=list(
        "amount"=function(n, x){rexp(n=n, rate=1/x)},
        "interact"=function(n, x){rbinom(n=n, size=1, p=x)}),
    CountSim=function(n, lambda){rpois(n, lambda)},  #remove the 1
    bucketNum=50,
    label=NULL,
    addRandomEffects=TRUE,
    parallel=FALSE,
    inputLog="expt with impact on user appearance depending on predictors;\n") {

  ## we record the input used to create data in a long string
  inputLog = paste(inputLog, "; \n", "userNum: ", userNum, "\n")
  inputLog = paste(
      "userDistbns:", paste(userDistbns, collapse="\n"),
      "mus:", paste(mus, collapse="--"), "\n",
      "valueCols_covars: ", paste(valueCols_covars, collapse="\n"), "\n",
      "transfList: ", paste(transfList, collapse="--"), "\n",
      "valueDistbns: ", paste(valueDistbns, collapse="--"), "\n",
      "addRandomEffects: ", addRandomEffects)

  userDf = SimUserAttr(
    userNum=userNum,
    distbns=userDistbns,
    balanceCol=NULL)

  ## add counterfactuals
  userDf = AddCounterfactual(df=userDf, exptCol="expt_id")[["fullDf"]]
  ## add a column to track expt id and counterfactuals
  userDf[ , "expt_id_cfactual"] = paste0(
      userDf[ , "expt_id"],
      "_",
      userDf[ , "cfactual"])

  userDt = data.table(userDf)
  userSummaryDt = userDt[ ,
      .(row_count=.N, user_count=length(unique(user_id))),
      by=c("expt_id", "cfactual")]


  valueDf = userDf
  for (i in 1:length(valueCols)) {
    valueCol = valueCols[i]

    valueDf = GenReg_linearPred(
      df=valueDf,
      mu0=mus[i],
      covars = valueCols_covars[[valueCol]],
      valueCol=valueCol)
  }

  ## this is just to test if counterfactual is set up correctly
  TestCf = function(valueDf, col, main) {
    contDf = valueDf[valueDf[ , "expt_id"] == "cont", ]
    exptDf = valueDf[valueDf[ , "expt_id"] == "treat", ]
    contDf = contDf[order(contDf[ , "user_id"]), ]
    exptDf = exptDf[order(exptDf[ , "user_id"]), ]
    df0 = merge(
        contDf[ , c("user_id", col)],
        exptDf[ , c("user_id", col)],
        by="user_id")
    plot(df0[ , paste0(col, ".x")], df0[ , paste0(col, ".y")], main=main)
  }

  userDf_re = valueDf
  if (addRandomEffects) {
    userDf_re = AddUser_randomEffects(
      userDf=valueDf,
      userCol="user_id",
      valueCols=valueCols,
      userVarCov_chol=userVarCov_chol,
      userVarCov=NULL)
  }

  userDf_re_trans = TransfCols(
      df=userDf_re,
      transfList=transfList)

  usageDf = GenUsageDf(
      userDf=userDf_re_trans,
      valueDistbns=valueDistbns,
      timestamp1=timestamp1,
      timestamp2=timestamp2,
      noTimestamps=noTimestamps,
      parallel=parallel)

  if (!is.null(label)) {
    userDf_re_trans[ , label[1]] = label[2]
    usageDf[ , label[1]] = label[2]
  }

  usageDf = usageDf[
      order(usageDf[ , "cfactual"], decreasing=TRUE),
      c("user_id", "country", "gender", "expt_id", "cfactual",
        "expt_id_cfactual", "timestamp", "obs_amount", "obs_interact")]

  Mod = GenModFcn(bucketNum)
  usageDf[ , "bucket"] = Mod(usageDf[ , "user_id"])
  usageDf[ , "date"] = format(
  as.Date(usageDf[ , "timestamp"], format="%Y-%m-%d"), "%Y-%m-%d")

  # we add a dummy column to easily count occurrences with sum below
  usageDf[ , "imp_count"] = 1

  usageDf2 = usageDf
  usageDf2[ , "timestamp"] = as.character(usageDf2[ , "timestamp"])
  usageDf2[ , "bucket"] = as.character(usageDf2[ , "bucket"])
  usageDf2[ , "imp_count"] = as.character(usageDf2[ , "imp_count"])
  usageDf2 = usageDf2[ ,  !(names(usageDf2) %in% "timestamp")]

  usageDt = data.table(usageDf)
  usageDtSummary = usageDt[ ,
      .(usage_count=.N, user_count=length(unique(user_id))),
      by=c("expt_id", "cfactual")]

  userDf2 = userDf_re_trans

  names(userDf2) = c(
      "user_id", "country", "gender", "expt_id", "cfactual",
      "expt_id_cfactual", "imp_count", "obs_amount", "obs_interact")

  ## assign the expected values for amount, interact
  userDf2[ , "obs_amount"] = userDf2[ , "obs_amount"] * userDf2[ , "imp_count"]
  userDf2[ , "obs_interact"] = userDf2[ , "obs_interact"] * userDf2[ , "imp_count"]

  userDt_usageConsisCols = data.table(userDf2)
  usageDt = data.table(usageDf)
  usageDt_obs = usageDt[get("cfactual") %in% "factual"]

  valueCols=c("imp_count", "obs_interact", "obs_amount")
  predCols = c("country", "gender")
  userDt_fromUsage_obs = DtSimpleAgg(
      dt=usageDt_obs,
      gbCols=c("user_id", predCols, "expt_id"),
      valueCols=valueCols,
      F=sum)

  userDt_withCf = DtSimpleAgg(
      dt=usageDt,
      gbCols=c("user_id", predCols, "expt_id", "cfactual", "expt_id_cfactual"),
      valueCols=valueCols,
      F=sum)

  return(list(
      "inputLog"=inputLog,
      "userDf"=userDf_re_trans,
      "usageDt"=usageDt,
      "usageDt_obs"=usageDt_obs,
      "userDt_usageConsisCols"=userDt_usageConsisCols,
      "userDt_fromUsage_obs"=userDt_fromUsage_obs,
      "userDt_withCf"=userDt_withCf,
      "usageDtSummary"=usageDtSummary))
}

## simulates data and generates plots
SimUsage_checkResults = function(
    userNum,
    predCols=c("country", "gender"),
    valueCols=c("imp_count", "obs_interact", "obs_amount"),
    parallel=FALSE) {

  usageData = SimUsageDf_e2e(
      userNum=userNum,
      noTimestamps=TRUE,
      label=c("product", "search"),
      bucketNum=50,
      parallel=parallel)

  inputLog = usageData[["inputLog"]]
  usageDt = usageData[["usageDt"]]
  usageDt_obs = usageData[["usageDt_obs"]]
  userDf = usageData[["userDf"]]
  userDt = data.table(userDf)
  userDt_usageConsisCols = usageData[["userDt_usageConsisCols"]]
  userDt_usageConsisCols_fac = userDt_usageConsisCols[get("cfactual") %in% "factual"]
  userDt_fromUsage_obs = usageData[["userDt_fromUsage_obs"]]
  userDt_withCf = usageData[["userDt_withCf"]]

  res = AddCounterfactual(
      df=userDt_fromUsage_obs, exptCol="expt_id")

  userDf_fromUsage_fac = res[["obsDf"]]
  userDf_fromUsage_cf = res[["cfDf"]]
  userDf_fromUsage_withCf = res[["fullDf"]]
  userDt_fromUsage_fac = data.table(userDf_fromUsage_fac)
  userDt_fromUsage_cf  = data.table(userDf_fromUsage_cf)

  ## here we do some validation of data
  # we calculate slice aggregates from user data
  # and from usage data
  userDfBased_sliceAgg = DtSimpleAgg(
      dt=userDt_usageConsisCols_fac,
      gbCols=c(predCols, "expt_id"),
      valueCols=valueCols)

  usageDfBased_sliceAgg = DtSimpleAgg(
      dt=userDt_fromUsage_obs,
      gbCols=c(predCols, "expt_id"),
      valueCols=valueCols)

  mdf = merge(
      userDfBased_sliceAgg,
      usageDfBased_sliceAgg,
      by=c(predCols, "expt_id"))

  pltList1 = list()
  for (col in valueCols) {
    x = paste0(col, ".x")
    y = paste0(col, ".y")
    a = mdf[ , get(x)]
    b = mdf[ , get(y)]
    df = data.frame(a=a, b=b)
    pltList1[[col]] = (
        ggplot(df, aes(x=a, y=b)) + geom_point(color=ColAlpha('blue', 0.5)) +
        xlab(paste0(col, ": userBased")) + ylab(paste0(col, ": usageBased")) +
        geom_abline(intercept=0, slope=1, color="red"))
  }

  Multiplot(pltList=pltList1, ncol=3)

  userDf_fromUsage_fac = Concat_stringColsDf(
      df=userDf_fromUsage_fac,
      cols=predCols,
      colName="slice", sepStr="-")

  BoxPlotIt = function(sliceCol, valueCol) {
    p = (ggplot(
          userDf_fromUsage_fac,
          aes(x=get(sliceCol), y=get(valueCol), fill=get("expt_id"))) +
        geom_boxplot() + ylab(valueCol) + xlab(sliceCol)) +
        guides(fill=guide_legend(title=valueCol)) +
        theme(axis.text.x = element_text(angle=15, hjust=1))

    return(p)
  }

  pltList2 = list()
  sliceCols= c(predCols, "slice")
  for (i in 1:length(sliceCols)) {
    for (j in 1:length(valueCols)) {
        sliceCol = sliceCols[i]
        valueCol = valueCols[j]
        key = paste0(sliceCol, "-", valueCol)
        pltList2[[key]] = BoxPlotIt(sliceCol=sliceCol, valueCol=valueCol)
    }
  }

  Multiplot(pltList=pltList2, ncol=length(valueCols))
  #userDt_cf = data.table(userDf_cf)

  return(list(
      "inputLog"=inputLog,
      "userDt_usageConsisCols"=userDt_usageConsisCols,
      "usageDt"=usageDt,
      "usageDt_obs"=usageDt_obs,
      "userDt_fromUsage_obs"=userDt_fromUsage_obs,
      "userDt_withCf"=userDt_withCf,
      "sliceBoxPltList"=pltList2
      ))
}

## write sim data
WriteSimData = function(
    ver,
    usageDt,
    userDt_usageConsisCols,
    userDt_fromUsage_obs,
    userDt_withCf,
    inputLog,
    dataPath) {

  #Mark(inputLog, "inputLog from inside WriteSimData")
  fn1 = file(paste0(dataPath, "usageDt", "_", ver, ".csv"), "w")
  write.csv(file=fn1,  x=data.frame(usageDt), row.names=FALSE)
  Mark(fn1, "written file location")
  close(fn1)

  fn2 = file(paste0(dataPath, "userDt_usageConsisCols", "_", ver, ".csv"), "w")
  write.csv(file=fn2,  x=data.frame(userDt_usageConsisCols), row.names=FALSE)
  close(fn2)

  fn3 = file(paste0(dataPath, "userDt_fromUsage_obs", "_", ver, ".csv"), "w")
  write.csv(file=fn3,  x=data.frame(userDt_fromUsage_obs), row.names=FALSE)
  close(fn3)

  fn4 = file(paste0(dataPath, "userDt_withCf", "_", ver, ".csv"), "w")
  write.csv(file=fn4,  x=data.frame(userDt_withCf), row.names=FALSE)
  close(fn4)

  fn5 = file(paste0(dataPath, "input_log_", ver, ".txt"), "w")
  cat(x=inputLog, file=fn5)
  close(fn5)
}

## sim data and write, the whole process
SimData_write = function(
    ver, parallel, userNum, dataPath) {

  simData = SimUsage_checkResults(
      userNum=userNum,
      predCols=c("country", "gender"),
      valueCols=c("imp_count", "obs_interact", "obs_amount"),
      parallel=parallel)

  userDt_usageConsisCols = simData[["userDt_usageConsisCols"]]
  usageDt = simData[["usageDt"]]
  usageDt_obs = simData[["usageDt_obs"]]
  userDt_fromUsage_obs = simData[["userDt_fromUsage_obs"]]
  userDt_withCf = simData[["userDt_withCf"]]
  inputLog = simData[["inputLog"]]

  #Mark(inputLog, "inputLog from SimData_write")
  WriteSimData(
      ver=ver,
      usageDt=usageDt,
      userDt_usageConsisCols=usageDt_obs,
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      userDt_withCf=userDt_withCf,
      inputLog=inputLog,
      dataPath=dataPath)

  # write it to public data folder
  fn0 = paste0(publicDataPath, "userDt_fromUsage_obs", ver, ".csv")
  fn = file(fn0, "w")
  write.csv(x=userDt_fromUsage_obs, file=fn)
  close(fn)

  gbCols = c("country", "gender", "expt_id")
  Check_forImbalance(dt=userDt_fromUsage_obs, predCols=c("country", "gender"))
}

## read sim data
ReadSimData = function(
    ver,
    dataPath,
    ReadF=read.csv,
    readInputLog=FALSE) {

  fn1 = file(paste0(dataPath, "usageDt", "_", ver, ".csv"), "r")
  usageDt = data.table(ReadF(file=fn1))
  close(fn1)

  fn2 = file(paste0(dataPath, "userDt_usageConsisCols", "_", ver, ".csv"), "r")
  userDt_usageConsisCols = data.table(ReadF(file=fn2))
  close(fn2)

  fn3 = file(paste0(dataPath, "userDt_fromUsage_obs", "_", ver, ".csv"), "r")
  userDt_fromUsage_obs = data.table(ReadF(file=fn3))
  close(fn3)

  fn4 = file(paste0(dataPath, "userDt_withCf", "_", ver, ".csv"), "r")
  userDt_withCf = data.table(ReadF(file=fn4))
  close(fn4)

  inputLog = NULL
  if (readInputLog) {
    fn5 = file(paste0(dataPath, "input_log_", ver, ".txt"), "r")
    inputLog = readLines(fn5)
    close(fn5)
  }

  return(list(
      "inputLog"=inputLog,
      "usageDt"=usageDt,
      "userDt_usageConsisCols"=userDt_usageConsisCols,
      "userDt_fromUsage_obs"=userDt_fromUsage_obs,
      "userDt_withCf"=userDt_withCf))
}

## checking the convergence of the experiment effect
# using various methods
CheckConverg = function(
    userNum,
    userDt_usageConsisCols,
    userDt_withCf,
    userDt_fromUsage_obs,
    valueCols,
    predCols,
    Diff,
    AggF,
    CommonMetric,
    metricList=NULL,
    gridNum=100) {

  GenPointEstimFcn = function(
    userDt,
    n,
    compareCol,
    comparePair) {

  function(n) {
    userSample = sample(1:userNum, n)
    userDt2 = userDt[user_id %in% userSample]

    outDf = CalcDiffMetrics_userDt(
        userDt=userDt2,
        compareCol=compareCol,
        comparePair=comparePair,
        valueCols=valueCols,
        Diff=Diff,
        AggF=AggF)

    outDf[1, "ss"] = n

    return(outDf)
    }

  }

  ## this is for adj version
  PointEstim = function(n) {

    #dt = usageDt[get("cfactual") %in% "factual"]
    compareCol = "expt_id"
    comparePair = c("treat", "cont")
    userSample = sample(1:userNum, n)
    #dt2 = dt[user_id %in% userSample]

    userDt_fromUsage_obs2 = userDt_fromUsage_obs[user_id %in% userSample]

    adjDiffList = CalcAdjMetrics_fromUserDt(
        userDt_fromUsage_obs=userDt_fromUsage_obs2,
        predCols=predCols,
        valueCols=valueCols,
        CommonMetric=CommonMetric,
        metricList=metricList)

    return(adjDiffList)
  }

  step = round(userNum / gridNum)
  init = max(c(step, 1000))
  x =  c(500, 750, seq(init, userNum, by=step))
  #Mark(x, "grid")

  G_user = GenPointEstimFcn(
      userDt=userDt_usageConsisCols,
      compareCol="expt_id",
      comparePair=c("treat", "cont"))


  ## first way to claculate the raw metrics
  G_obs =   G_adj_contDataOnly = function(n) {
    return(PointEstim(n)[["raw"]])
  }

  ## second way to calculate
  G_obs2 = GenPointEstimFcn(
    userDt=userDt_withCf[get("cfactual") %in% "factual"],
    compareCol="expt_id",
    comparePair=c("treat", "cont"))


  G_adj_contDataOnly = function(n) {
    return(PointEstim(n)[["adjDiff_contDataOnly"]])
  }

  G_adj_withTreatData = function(n) {
    return(PointEstim(n)[["adjDiff_withTreatData"]])
  }

  G_cfDiff_expt = GenPointEstimFcn(
      userDt=userDt_withCf[
          get("expt_id_cfactual") %in% c("treat_factual", "cont_cfactual")],
      compareCol="expt_id_cfactual",
      comparePair=c("treat_factual", "cont_cfactual"))

  G_cfDiff_cont = GenPointEstimFcn(
      userDt=userDt_withCf[
          get("expt_id_cfactual") %in% c("treat_cfactual", "cont_factual")],
      compareCol="expt_id_cfactual",
      comparePair=c("treat_cfactual", "cont_factual"))

  fcnList = list(
      "userBasedDiff"=G_user,
      "obsDiff"=G_obs,
      "obsDiff2"=G_obs2,
      "adjDiff_contDataOnly"=G_adj_contDataOnly,
      "adjDiff_withTreatData"=G_adj_withTreatData,
      "cfDiff_expt"=G_cfDiff_expt,
      "cfDiff_cont"=G_cfDiff_cont)

  methods = names(fcnList)

  F = function(method) {
    df0 = do.call(rbind, lapply(x, FUN=fcnList[[method]]))
    if (!("ss" %in% names(df0))) {
      df0[ , "ss"] = x
    }
    return(df0)
  }

  dtList = lapply(methods, FUN=F)
  names(dtList) = methods

  estimDf = setNames(
        data.frame(matrix(ncol=length(valueCols) + 2, nrow=0)),
        c("method", "ss", valueCols))

  estimDt = data.table(estimDf)

  colSuffix = c(
      rep("treat_vs_cont", 5),
      "treat_factual_vs_cont_cfactual", "treat_cfactual_vs_cont_factual")

  for (i in 1:length(methods)) {

    method = methods[i]
    cols = paste0(valueCols, "_", colSuffix[i])
    df0 = dtList[[method]]
    dt0 = data.table(df0)

    dt0[ , "method"] = method

    dt0 = dt0[ , mget(c("method", "ss", cols))]
    names(dt0) = c("method", "ss", valueCols)
    estimDt = rbind(estimDt, dt0)
  }


  Plt = function() {
    for (col in valueCols) {
      userBasedCol = paste0(col, "_", colSuffix[1])
      obsCol = paste0(col, "_", colSuffix[2])
      adjCol = paste0(col, "_", colSuffix[3])
      cfCol_expt = paste0(col, "_", colSuffix[4])
      cfCol_cont = paste0(col, "_", colSuffix[5])

      yMin = min(c(
          data.frame(dtList[["obsDiff"]])[ , obsCol],
          data.frame(dtList[["adjDiff_contDataOnly"]])[ , adjCol],
          data.frame(dtList[["adjDiff_withTreatData"]])[ , adjCol],
          data.frame(dtList[["userBasedDiff"]])[ , userBasedCol]))

      yMax = max(c(
          data.frame(dtList[["obsDiff"]])[ , obsCol],
          data.frame(dtList[["adjDiff_contDataOnly"]])[ , adjCol],
          data.frame(dtList[["adjDiff_withTreatData"]])[ , adjCol],
          data.frame(dtList[["userBasedDiff"]])[ , userBasedCol]))

      plot(
          1:nrow(dtList[["userBasedDiff"]]),
          data.frame(dtList[["userBasedDiff"]])[ , userBasedCol],
          col=ColAlpha("black", 1), type="l", lwd=2, main=col,
          ylim=c(yMin, yMax), xlab=paste0("user num in ", step, "s"), ylab=col)

      #lines(
      #    1:nrow(cfDiff_expt), data.frame(cfDiff_expt)[ , cfCol_expt],
      #    col=ColAlpha("green", 0.4), lwd=2)

      #lines(
      #    1:nrow(cfDiff_cont), data.frame(cfDiff_cont)[ , cfCol_cont],
      #    col=ColAlpha("lightgreen", 0.4), lwd=2)

      lines(
          1:nrow(dtList[["obsDiff"]]),
          data.frame(dtList[["obsDiff"]])[ , obsCol],
          col=ColAlpha("red", 0.4), lwd=2)

      lines(
          1:nrow(dtList[["adjDiff_contDataOnly"]]),
          data.frame(dtList[["adjDiff_contDataOnly"]])[ , adjCol],
          col=ColAlpha("blue", 0.4), lwd=2)

      lines(
          1:nrow(dtList[["adjDiff_withTreatData"]]),
          data.frame(dtList[["adjDiff_withTreatData"]])[ , adjCol],
          col=ColAlpha("lightblue", 0.2), lwd=2)

      legend(
          x=nrow(dtList[["obsDiff"]])*(3/4), y=yMax,
          legend=c(
            "user_based", "treat_f/cont_cf", "treat_cf/cont_f", "obs",
            "adj_contDataOnly", "adj_withTreatData"),
          col=c("black", "green", "lightgreen", "red", "blue", "lightblue"),
          lwd=2, lty=1, cex=0.8)
    }
  }

  return(list("dtList"=dtList, "estimDt"=estimDt))
}

## check convergence for this use case
CheckConverg_useCase = function(AggF, CommonMetric, userNum) {

  out = CheckConverg(
    userNum=userNum,
    userDt_usageConsisCols=userDt_usageConsisCols,
    userDt_withCf=userDt_withCf,
    userDt_fromUsage_obs=userDt_fromUsage_obs,
    valueCols=c("obs_amount", "obs_interact", "imp_count"),
    predCols=c("gender", "country"),
    Diff=function(x, y) {x / y},
    AggF=AggF,
    CommonMetric=CommonMetric,
    metricList=NULL,
    gridNum=5)

  return(out[["estimDt"]])
}

## plotting the results of metric convg, by calculating mean, sd per
# sample size and method
PltEstimDt_meanSdConvg = function(
    estimDt, valueCols, methods=NULL, ssUpperLim=NULL, titleSuffix="",
    sizeAlpha=1) {

  meanDt = DtSimpleAgg(dt=estimDt, gbCols=c("ss", "method"), F=mean)
  sdDt = DtSimpleAgg(dt=estimDt, gbCols=c("ss", "method"), F=sd)

  if (is.null(methods)) {
    methods = unique(estimDt[ , method])
    Mark(methods)
  }

  meanDt = meanDt[method %in% methods ]
  sdDt = sdDt[method %in% methods]

  if (!is.null(ssUpperLim)) {
    meanDt = meanDt[ss < ssUpperLim]
    sdDt = sdDt[ss < ssUpperLim]
  }

  Plt = function(col) {

    lwd = 1.5
    alpha = 0.75
    size = 22
    pltList = list()
    pltList[[paste0("mean_", col)]] = (
        ggplot(meanDt, aes(x=ss, y=get(col), fill=method)) +
        geom_line(
            aes(color=method, linetype=method), alpha=alpha, size=lwd*sizeAlpha) +
        ggtitle(paste0("mean; ", titleSuffix)) +
        xlab("user_num") +
        ylab(paste0("mean of ", col))) +
        scale_y_continuous(limits=c(0.5, 2.5)) +
        theme(
            text=element_text(size=size*sizeAlpha),
            axis.text.x=element_text(angle=90, hjust=1))

    pltList[[paste0("sd_", col)]] = (
        ggplot(sdDt, aes(x=ss, y=get(col), fill=method)) +
        geom_line(
            aes(color=method, linetype=method), alpha=alpha, size=lwd*sizeAlpha) +
        ggtitle(paste0("sd; ", titleSuffix)) +
        xlab("user_num") +
        ylab(paste0("sd of ", col))) +
        theme(
            text=element_text(size=size*sizeAlpha),
            axis.text.x=element_text(angle=90, hjust=1))

    return(pltList)
  }

  pltList = NULL
  for (col in valueCols) {
    pltList = c(pltList, Plt(col))
  }

  return(pltList)
}

## opens the simulation data generated by the simulation job:
GetEstimDt = function(ver, metricName, parallel, convgDataPath) {
  path = paste0(
      convgDataPath,
      ver, "/", metricName, "/")

  dfList = OpenDataFiles(path, patterns=NULL, parallel=parallel)

  estimDt = do.call(rbind, dfList)
  estimDt = data.table(estimDt)
  return(estimDt)
}

## this will plot the mean and sd of the estimators for various methods
# as a function of the sample size
# also it will also compares the sd of other methods with raw
PltAndSave_simResults = function(
  ver, metricName, parallel, convgDataPath) {

  estimDt = GetEstimDt(
      ver=ver, metricName=metricName, parallel=parallel,
      convgDataPath=convgDataPath)


  ## plotting convg results
  estimDt2 = estimDt[ss < 10000]
  estimDt2 = estimDt2[ss > 1000]
  methods = c("obsDiff", "adjDiff_contDataOnly", "adjDiff_withExptData")
  #methods = c("obsDiff", "adjDiff_contDataOnly", "adjDiff_withTreatData")
  valueCols = c("imp_count", "obs_interact", "obs_amount")
  valueCols2 = c("imp", "interact", "amount")
  colnames(estimDt2) = mapvalues(
      colnames(estimDt2),
      from=valueCols,
      to=valueCols2)
  #methods = c("obsDiff", "adjDiff_contDataOnly", "adjDiff_withTreatData", "cfDiff_expt", "cfDiff_cont")

  estimDt2 = DtRemap_colValues(
      dt=estimDt2,
      col="method",
      values=methods,
      newValues=c("raw", "adj_cont", "adj_all"))

  methods = c("raw", "adj_cont", "adj_all")

  r = 1
  convgPltList = PltEstimDt_meanSdConvg(
      estimDt=estimDt2,
      valueCols=valueCols2,
      methods=methods,
      ssUpperLim=NULL,
      titleSuffix=metricName,
      sizeAlpha=r)

  r = 0.9
  Device=function(...) {
          Cairo::CairoPNG(..., units="in", dpi=100*r, pointsize=20*r)
  }

  pltList = convgPltList
  PltOne = function(name) {
    fn = paste0(figsPath, metricName, "_", name, "_", ver, ".png")
    print(fn)
    fn = file(fn,  open='w')
    p = pltList[[name]]
    #if (startsWith(name, "sd")) {
    #  p = p + guides(fill=FALSE)
    #} else {
    #  p = p + theme(axis.text.y=element_blank())
    #}
    #print(p)
    ggsave(filename=fn, plot=p, device=Device, width=6*r,  height=6*r)
    dev.off()
    close(fn)
  }

  lapply(X=names(pltList), FUN=PltOne)


  r = 2.5
  size1 = 25
  size2 = 20
  pltList = convgPltList
  pltList = lapply(
      X=pltList,
      FUN=function(x) {
        return(
            x + theme(
                axis.text=element_text(size=size2*r),
                axis.title=element_text(size=size2*r),
                plot.title=element_text(size=size1*r),
                legend.title=element_text(size=size2*r),
                legend.text=element_text(size=size2*r)))
      })

  Device=function(...) {
          Cairo::CairoPNG(..., units="in", dpi=120*r, pointsize=12*r)
  }

  fn0 = paste0(figsPath, metricName, "_", ver, ".png")
  Mark(fn0, "filename")
  fn = file(fn0,  open='w')

  GgsaveMulti(
      fn=fn,
      pltList=pltList,
      ncol=2,
      Device=Device,
      width=6*r,
      height=6*r)


  PltConvg = function() {
      Multiplot(pltList=convgPltList, ncol=2)
  }

  compareSdRes = CompareMethodsSd(
      estimDt=estimDt2,
      methods=methods,
      valueCols=valueCols2,
       mainSuffix=metricName)

  PltCompareSd = compareSdRes[["Plt"]]

  fn0 = paste0(figsPath, metricName, "_sd_comparison_", ver, ".png")
  Mark(fn0, "filename")
  fn = file(fn0, "w")

  r = 3
  Cairo(
      width=640*r, height=480*r, file=fn, type="png", dpi=120*r,
      pointsize=8*r)
  PltCompareSd()
  dev.off()
  close(fn)

  #CompareSdPlt()
  return(list(
      "estimDt"=estimDt,
      "convgPltList"=convgPltList,
      "PltConvg"=PltConvg,
      "PltCompareSd"=PltCompareSd))
}

PltAndSave_ciConvRes = function(
    ver, metricName, parallel,
    compareValues=c("raw", "control_data", "all_data")) {

  dt = copy(userDt_fromUsage_obs)
  metricDict = list(
      "mean_ratio"=Metric_meanRatio, "sum_ratio"=Metric_sumRatio)
  CommonMetric = metricDict[[metricName]]

  predCols = c("country", "gender")
  valueCols = c("obs_amount", "obs_interact", "imp_count")

  res = CiLengthConvg(
      dt=dt, gridNum=40, valueCols=valueCols, predCols=predCols,
      CommonMetric=CommonMetric, bs=FALSE, bsNum=300,
      compareValues=compareValues, userNumProp=1/20,
      parallel=parallel, mainSuffix=metricName)


  fn0 = paste0(figsPath, metricName, "_ci_convg_comparison_", ver, ".png")
  Mark(fn0, "filename")
  fn = file(fn0, "w")

  r = 3
  Cairo(
      width=640*r, height=480*r, file=fn, type="png", dpi=120*r,
      pointsize=8*r)
  res[["Plt"]](res[["jkDf"]])
  dev.off()
  close(fn)
  return(res)
}

## calc final metrics for the simulations
CalcFinalMetricsCi = function(
    ss, parallel=FALSE, CommonMetric=Metric_meanRatio,
    predCols=c("country", "gender"),
    valueCols=c("obs_amount", "obs_interact", "imp_count")) {

  dt = copy(userDt_fromUsage_obs)
  users = unique(dt[ , user_id])
  print(length(users))
  userSample = sample(users, ss)
  dt = dt[user_id %in% userSample]
  length(dt[ , user_id])
  nrow(dt)
  Mod = GenModFcn(50)
  dt[ , "bucket"] = Mod(as.numeric(dt[ , user_id]))

  ciDf_simple = CalcMetricCis_withBuckets(
      dt=dt, valueCols=valueCols, predCols=predCols, CommonMetric=CommonMetric,
      metricList=NULL, ci_method="simple_bucket", parallel=parallel)

  ciDf_jk = CalcMetricCis_withBuckets(
      dt=dt, valueCols=valueCols, predCols=predCols, CommonMetric=CommonMetric,
      ci_method="jk_bucket", parallel=parallel)

  ciDf_bs = CalcMetricCis_withBootstrap(
      dt=dt, valueCols=valueCols, predCols=predCols,
      CommonMetric=CommonMetric, parallel=parallel)

  ciDf_simple = TidyCiDf(ciDf_simple)
  ciDf_jk = TidyCiDf(ciDf_jk)
  ciDf_bs = TidyCiDf(ciDf_bs)

  Mark(ciDf_simple)
  Mark(ciDf_jk)
  Mark(ciDf_bs, 3)

  return(list(
      "ciDf_simple"=ciDf_simple,
      "ciDf_jk"=ciDf_jk,
      "ciDf_bs"=ciDf_bs))
}

## does jackknife CIs and compares with raw in a colored latex table
CompareExptCi_latex = function(
    dt, metricName, predCols, valueCols,
    parallel, ci_method="jk_bucket", ss=NULL,
    writePath, fnSuffix,
    maxCoreNum=NULL) {

  dt2 = copy(dt)
  if (!is.null(ss)) {
    ss = min(ss, nrow(dt))
    dt2 = dt[sample(.N, ss)]
  } else {
    ss = nrow(dt)
  }

  metricDict = list(
      "mean_ratio"=Metric_meanRatio, "sum_ratio"=Metric_sumRatio)
  CommonMetric = metricDict[[metricName]]

  Mod = GenModFcn(50)
  dt2[ , "bucket"] = Mod(as.numeric(dt2[ , user_id]))
  res = CalcMetricCis_withBuckets(
      dt=dt2, valueCols=valueCols, predCols=predCols,
      CommonMetric=CommonMetric,
      ci_method=ci_method, parallel=parallel, maxCoreNum=maxCoreNum)

  ciDf = res[["ciDf"]]
  ciDf = RoundDf(ciDf, 4)
  rownames(ciDf) = NULL

  ciDf = StarCiDf(
      df=ciDf, upperCol="ci_upper", lowerCol="ci_lower",
      upperThresh=c(1, 1.5, 2), lowerThresh=c(1, 0.75, 0.5))

  ss_str = ""
  if (ss < 1000) {
    ss_str = ss
  } else if (ss >= 1000 & ss < 10^6) {
    ss_str = paste0(floor(ss / 1000), "k")
  } else {
    ss_str = paste0(round(ss / 10^6, 1), "m")
  }

  ciDf[ , "resp"] = gsub("_treat_vs_cont", "", ciDf[ , "resp"])
  ciDf[ , "method"] = gsub(
      "adjDiff_contDataOnly", "adj_cont_data", ciDf[ , "method"])
  ciDf[ , "method"] = gsub(
      "adjDiff_withTreatData", "adj", ciDf[ , "method"])
  ciDf[ , "ci_method"] = gsub("_bucket", "", ciDf[ , "ci_method"])
  ciDf[ , "ss"] = ss_str
  ciDf = ciDf[ciDf[ , "method"] %in% c("raw", "adj"), ]
  ciDf = DfSubsetCols(ciDf, dropCols=c("ci_method"))

  fn0 = paste0(
        "ci_comparison_", metricName,
        "_ss_", ss_str, "_", fnSuffix)

  caption = gsub(x=fn0, "_", " ")
  label = fn0
  fn0 = paste0(writePath, fn0, ".tex")
  fn0 = tolower(fn0)
  print(fn0)
  fn = file(fn0, "w")

  colnames(ciDf) = gsub("_", ".", x=colnames(ciDf))

  for (col in colnames(ciDf)) {
    ciDf[ , col] = gsub("_", ".", x=as.character(ciDf[ , col]))
  }

  ind1 = ciDf[ , "method"] == "raw"
  ind2 = ciDf[ , "method"] == "adj"
  for (col in c("method", "ci.length")) {
    ciDf[ind1, col] = paste("\\color{red}", ciDf[ind1, col])
    ciDf[ind2, col] = paste("\\color{blue}", ciDf[ind2, col])
  }

  ciDf = ciDf[ , c(
      "resp", "method", "ci.lower", "ci.upper", "ci.length", "sig.stars")]
  colnames(ciDf) = c("resp", "method", "lower", "upper", "ci.length", "sig.")
  x = xtable2(ciDf, caption=caption, label=label,  digits=3)
  print(
       x=x, file=fn, include.rownames=FALSE,
       hline.after=c(-1, 0, 1:nrow(ciDf)),
       size="tiny",
       sanitize.text.function=identity)
  close(fn)

  return(list(
      "jkRes"=res,
      "ciDf"=ciDf,
      "latexTab"=x,
      "ss_str"=ss_str))
}

CompareExptCi_latex_slices = function(
    dt, metricName, predCols, valueCols, sliceCol,
    parallel, ci_method="jk_bucket", onlySigSlices=FALSE,
    ss=NULL, minSs=NULL, writePath, fnSuffix, maxCoreNum=NULL) {

  tab = table(df[ , sliceCol])
  if (!is.null(minSs)) {
    tab = tab[tab > minSs]
  }

  slices = names(tab)

  G = function(slice) {
    dt2 = dt[get(sliceCol) == slice]
    res = CompareExptCi_latex(
        dt=dt2, metricName="mean_ratio", valueCols=valueCols,
        predCols=predCols, parallel=parallel, ci_method=ci_method,
        ss=ss, writePath=tablesPath,
        fnSuffix=paste0(fnSuffix, "_", sliceCol, "_", slice))
    ciDf = res[["ciDf"]]
    ciDf[ , "sample.size"] = res[["ss_str"]]
    slice = gsub("_", ".", slice)
    slice = gsub("<", "less", slice)
    slice = gsub(">", "more", slice)
    ciDf[ , "slice"] = paste0(sliceCol, ".", slice)
    cols = c(
        "resp", "slice", "sample.size", "method",
        "lower", "upper", "ci.length", "sig.")

    ciDf = ciDf[ , cols]
    return(ciDf)
  }

  ciDf = do.call(what=rbind, args=lapply(X=slices, FUN=G))

  if (onlySigSlices) {
    ind = ciDf[ , "sig."] != ""
    if (length(ind) > 0) {
      sigSlices = ciDf[ind, "slice"]
      ciDf = ciDf[ciDf[ , "slice"] %in% sigSlices, ]
    }
  }

  colnames(ciDf) = gsub("_", ".", colnames(ciDf))
  ciDf[ , "slice"] = gsub("_", ".", ciDf[ , "slice"])
  ciDf[ , "slice"] = gsub("<", "less", ciDf[ , "slice"])

  fn0 = paste0(
      "ci_comparison_", metricName, "_", fnSuffix, "_", sliceCol)

  caption = gsub(x=fn0, "_", " ")
  label = fn0
  fn0 = paste0(writePath, fn0, ".tex")
  fn0 = tolower(fn0)
  print(fn0)
  fn = file(fn0, "w")

  x = xtable2(ciDf, caption=caption, label=label, digits=4)
  print(
       x=x, file=fn, include.rownames=FALSE,
       hline.after=c(-1, 0, 1:nrow(ciDf)),
       size="tiny",
       sanitize.text.function=identity)
  close(fn)

  return(list(
      "ciDf"=ciDf,
      "latexTab"=x
      ))
}

## write covariate performance
Write_covariatePerf = function(
    predCols, valueCols, ss=10^5, proj, exptId, label="", tablesPath="") {

  k = min(ss, nrow(dt))
  df = data.frame(dt[sample(.N, k)])
  res = FitPred_multi(
      df=df,
      newDf=df,
      valueCols=valueCols,
      predCols=predCols,
      commonFamily=gaussian,
      predQuantLimits=c(0, 1))

  # "theta", "sd_ratio"
  res = res[["infoDf"]][ , c("valueCol", "cv_cor")]

  CiLengthReduction = function(r) {
    1 - sqrt(1 - r^2)
  }

  res[ , "ci reduct"] = CiLengthReduction(res[ , "cv_cor"])

  res[ , "ss multiplier"] = VarReduct_sampleSizeGain(
      res[ , "cv_cor"])

  colnames(res)[1:2] = c(
      "response", "cv corr", "ss multiplier")

  caption = gsub("_", " ", label)
  caption = paste("Predictors:", caption)

  tab = xtable2(res, caption=caption, label=label)

  fn0 = label
  fn = paste0(tablesPath, proj, "_",  exptId, "_", fn0, ".tex")
  print(fn)
  fn = file(fn, "w")

  print(
      x=tab, file=fn, include.rownames=FALSE,
      hline.after=c(-1, 0, 1:nrow(tab)))

  close(fn)
  return(tab)
}

# opens a simulation version data
OpenData_Explore_simVer = function(ver, dataPath) {

  t1 = Sys.time()
  data = ReadSimData(
      ver=ver, ReadF=read.csv, readInputLog=TRUE, dataPath=dataPath)
  t2 = Sys.time()
  Mark(t2 - t1)

  valueCols = c("imp_count", "obs_interact", "obs_amount")
  predCols = c("gender", "country")
  Diff = function(x, y){x / y}
  usageDt = data[["usageDt"]]
  userDt_usageConsisCols = data[["userDt_usageConsisCols"]]
  userDt_fromUsage_obs = data[["userDt_fromUsage_obs"]]
  userDt_withCf = data[["userDt_withCf"]]
  inputLog = data[["inputLog"]]
  userNum = length(unique(userDt_usageConsisCols[ , user_id]))
  Mark(userNum)
  dt = userDt_fromUsage_obs
  p1 = Check_forImbalance(dt=dt, predCols=c("gender"))[["p"]]
  p2 = Check_forImbalance(dt=dt, predCols=c("country"))[["p"]]
  #p3 = Check_forImbalance(dt=dt, predCols=c("gender", "country"))[["p"]]
  pltList = list(p1, p2)
  fn0 = paste0(figsPath, "check_for_imbalance_", ver, ".png")
  print(fn0)
  fn = file(fn0, "w")

  Multiplot(pltList)
  inputLog

  GgsaveMulti(
      fn=fn,
      pltList=pltList)

  #@title check the number of users per arm
  ## check number of users per arm
  gbCols = c("country", "expt_id", "cfactual")
  gbCols =  c("expt_id", "cfactual")
  userCntDt = DtSimpleAgg(
      dt=usageDt,
      gbCols=gbCols,
      valueCols="user_id",
      F=function(x)length(unique(x)))
  #userCntDt = userCntDt[order(userCntDt[ , country]), ]
  Mark(userCntDt, "from usage data")
  userCntDt = DtSimpleAgg(
      dt=userDt_usageConsisCols,
      gbCols=gbCols,
      valueCols="user_id",
      F=function(x)length(unique(x)))
  #userCntDt = userCntDt[order(userCntDt[ , country]), ]
  Mark(userCntDt, "from underlying user data, before usage is simulated")

  #@title Assess prediction power and write latex table to file
  n = 10000
  n = min(n, userNum)
  userSample = sample(1:userNum, n)
  userDt_fromUsage_obs2 = userDt_fromUsage_obs[user_id %in% userSample]
  AssessPredPower_calcTheta(
      userDt=userDt_fromUsage_obs2,
      valueCols=valueCols,
      predCols=predCols,
      writeIt=TRUE,
      writePath=tablesPath,
      fnSuffix=ver)

  #@title check adjustment balance: h
  S()
  userNum = length(userDt_fromUsage_obs[ , user_id])

  userDt_fromUsage_obs[ , "amount"] = userDt_fromUsage_obs[ , "obs_amount"]
  userDt_fromUsage_obs[ , "interact"] =  userDt_fromUsage_obs[ , "obs_interact"]
  userDt_fromUsage_obs[ , "imp"] = userDt_fromUsage_obs[ , "imp_count"]

  ## checking balance
  res = Check_adjustmentBalance(
      userDt_fromUsage_obs=userDt_fromUsage_obs,
      predCols=c("country", "gender"),
      valueCols=c("amount", "interact", "imp"),
      Diff=function(x, y){x / y},
      ss=500,
      savePlt=TRUE,
      fnSuffix=ver)

  res[["Plt"]]()

  ####
  #@title Calculate final metrics for the simulations
  #res = CalcFinalMetricsCi(ss=5000, parallel=TRUE)

  ## check convergence of estimators "locally"
  G = function(x) {
    return(CheckConverg_useCase(
          AggF=mean, CommonMetric=Metric_meanRatio, userNum=round(userNum / 4)))
  }

  #estimDt = do.call(rbind, lapply(1:5, FUN=G))
  #Mark(dim(estimDt))
  return(list("userDt_fromUsage_obs"=userDt_fromUsage_obs))
}

Check_metricConvg_simVer = function(
    ver, metricName, userDt_fromUsage_obs, parallel=TRUE, convgDataPath) {

  # metricName = "sum_ratio"
  closeAllConnections()
  res = PltAndSave_simResults(
      ver=ver, metricName=metricName, parallel=parallel,
      convgDataPath=convgDataPath)
  PltConvg = res[["PltConvg"]]
  PltCompareSd = res[["PltCompareSd"]]

  PltConvg()
  PltCompareSd()

  #@title check convergence of CIs
  closeAllConnections()

  dt = copy(userDt_fromUsage_obs)
  colnames(dt) = mapvalues(
      colnames(dt),
      from=c("obs_amount", "obs_interact", "imp_count"),
      to=c("amount", "interact", "imp"))

  CommonMetric = Metric_meanRatio
  predCols = c("country", "gender")
  valueCols = c("amount", "interact", "imp")
  compareValues = c("raw", "control_data", "all_data")

  res = CiLengthConvg(
      dt=dt, gridNum=100, valueCols=valueCols, predCols=predCols,
      CommonMetric=CommonMetric, bs=FALSE, bsNum=300,
      compareValues=compareValues,
      userNumProp=1/5, parallel=parallel, parallel_outfile=parallel_outfile)

  res[["Plt"]](res[["jkDf"]])

  fn0 = paste0(figsPath, metricName, "_ci_convg_comparison_", ver, ".png")
  Mark(fn0, "filename")
  fn = file(fn0, "w")

  r = 3
  Cairo(
      width=640*r, height=480*r, file=fn, type="png", dpi=120*r,
      pointsize=8*r)
  res[["Plt"]](res[["jkDf"]])
  dev.off()
  close(fn)

  PltAndSave_ciConvRes(
      ver=ver, metricName=metricName,
      parallel=parallel,
      compareValues=compareValues)
}
