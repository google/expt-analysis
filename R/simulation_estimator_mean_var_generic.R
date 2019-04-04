## simulation study code

#library("ggplot2")
library("data.table")
library("data.analysis")
library("rglib")
library("tfruns")

FLAGS = flags(flag_numeric(("sim", "", "the sim number."))

readPath = "~/data/"
writePath = "~/data"
ver = "v1"

simData = ReadSimData(ver, dataPath=readPath)

ConvgDt = function(metricInfo) {

  valueCols = c("imp_count", "obs_interact", "obs_amount")
  predCols = c("gender", "country")
  usageDt = simData[["usageDt"]]
  userDt_usageConsisCols = simData[["userDt_usageConsisCols"]]
  userDt_fromUsage_obs = simData[["userDt_fromUsage_obs"]]
  userDt_withCf = simData[["userDt_withCf"]]
  userNum = length(unique(userDt_usageConsisCols[ , user_id]))
  #Mark(userNum)
  #res = CheckConverg_useCase(AggF=sum, CommonMetric=Metric_meanRatio, userNum=round(userNum/2))
  #return(res)

  out = CheckConverg(
    userNum=round(userNum) / 4,
    userDt_usageConsisCols=userDt_usageConsisCols,
    userDt_withCf=userDt_withCf,
    userDt_fromUsage_obs=userDt_fromUsage_obs,
    valueCols=c("obs_amount", "obs_interact", "imp_count"),
    predCols=c("gender", "country"),
    Diff=metricInfo[["Diff"]],
    AggF=metricInfo[["AggF"]],
    CommonMetric=metricInfo[["Metric"]],
    metricList=NULL,
    gridNum=150)

  dt = out[["estimDt"]]

  dt[ , "sim_num"] = FLAGS$sim

  fn = file(
      paste0(
          writePath, ver, "/", metricInfo[["name"]], "/estim_dt_",
          as.character(FLAGS$sim), "_", ver, ".csv"), "w")

  write.csv(file=fn,  x=dt, row.names=FALSE)
  close(fn)

}


metricInfo1 = list(
    "name"="mean_ratio", "Metric"=Metric_meanRatio, "AggF"=mean,
    "Diff"=function(x, y) {x / y})

metricInfo2 = list(
    "name"="sum_ratio", "Metric"=Metric_sumRatio, "AggF"=sum,
    "Diff"=function(x, y) {x / y})

res = lapply(X=list("mi1"=metricInfo1, "mi2"=metricInfo2), FUN=ConvgDt)
