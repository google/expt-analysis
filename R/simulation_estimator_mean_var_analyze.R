# author: rezani@
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



## This code is to analyze the results of simulations and
# is not needed for analyzing data

Src = function() {

  files=c(
      "data_analysis.R",
      "unit_analysis.R",
      "unit_analysis_use_case.R",
      "plotting.R",
      "clustering.R"))

  Library = function(x) {
    suppressMessages(library(x, character.only=TRUE))
  }

  try(Library("colorout"))
  Library("Cairo")
  Library("ggplot2")
  Library("data.table")
  Library("xtable")
  Library("gridExtra")
  Library("grid")
  Library("stringi")

  codePath = ""

  for (fn0 in files)
  {
    fn = file(paste0(codePath, fn0))
    source(fn)
    close(fn)
  }
}

Src()


## specify destinations for saving figs and tables
proj = "var_reduction"
path = paste0("", prod, "/")
publicDataPath = ""
figsPath = paste0(path, "figs/")
dataPath = paste0(path, "data/")
tablesPath = paste0(path, "tables/")

localPath = paste0("~/work/projects/", proj, "/")
parallel_outfile = paste0(localPath, "parallel_outfile.R")

## specify data paths
dataPath = ""
convgDataPath = ""

## Approximated reduction as a result of adjustment
res = Plt_adjCiLengthReduct(figsPath=figsPath)
res = Plt_ssCiLengthReduct(figsPath=figsPath)

## WRITE
# simulate data
S()
ver = "v17"
t1 = Sys.time()
##SimData_write(ver=ver, parallel=TRUE, userNum=10^5, dataPath=dataPath)
t2 = Sys.time()
Mark(t2 - t1)


## READ
# read simulated data
S()
ver = "v17"

res = OpenData_Explore_simVer(ver, dataPath=dataPath)
userDt_fromUsage_obs = res[["userDt_fromUsage_obs"]]

res = Check_metricConvg_simVer(
    ver=ver, metricName="mean_ratio",
    userDt_fromUsage_obs=userDt_fromUsage_obs,
    convgDataPath=convgDataPath)

## needed only for v15 in the paper.
res = Check_metricConvg_simVer(
    ver=ver, metricName="sum_ratio",
    userDt_fromUsage_obs=userDt_fromUsage_obs,
    convgDataPath=convgDataPath)


## Complex metrics
# check convergence of CIs
S()
closeAllConnections()
dt = copy(userDt_fromUsage_obs)

predCols = c("country", "gender")
valueCols = c("obs_amount", "obs_interact", "imp_count")
compareValues = c("raw", "adjDiff_contDataOnly", "adjDiff_withTreatData")
compareValues = c("raw", "control_data", "all_data")

bivarMetric = list(
    "F"=Metric_ratioOfMeanRatios,
    "col1"="obs_interact",
    "col2"="imp_count")

res = CiLengthConvg(
    dt=dt, gridNum=40, valueCols=valueCols, predCols=predCols,
    CommonMetric=NULL, bivarMetric=bivarMetric, bs=FALSE, bsNum=300,
    compareValues=compareValues,
    userNumProp=1/5, parallel=TRUE)

res[["Plt"]](res[["jkDf"]])

metricName = "ratio_of_mean_ratios"
fn0 = paste0(figsPath, metricName, "_ci_convg_comparison_", ver, ".png")
Mark(fn0, "filename")


fn = file(fn0, "w")

r = 1.5
Cairo(
    width=640*r, height=480*r, file=fn, type="png", dpi=120*r,
    pointsize=10*r)

res[["Plt"]](res[["jkDf"]])

dev.off()
close(fn)
