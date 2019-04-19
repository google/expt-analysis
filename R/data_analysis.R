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

## helper functions for R data analysis
# proprietary information

Mark = function(x, text="") {
  str = paste0(
      "\n *** object class: ",
      paste(as.character(class(data.table())), collapse=", "))
  if (text != "") {
    str = paste0(str, "; message: ", text)
  }
  cat(str, "\n")
  print(x)
}

TestMark = function() {
  x = 2
  Mark(x, text="this is the value of x")
}

# print a vector horizontally
Pr = function(x) {

  for (i in 1:length(x)) {
    print(x[i])
  }
}

## print columns of data frame in readable format
PrCols = function(df) {
  Pr(colnames(df))
}

## some basic info about data.frame
DescribeDf = function(df) {

  print(paste0(
      "this is the dim: ", "row_num: ", dim(df)[1], "; col_num: ", dim(df)[2]))
  colsStr = paste(colnames(df), collapse="--")
  print(paste("these are the columns:", colsStr))

  print("sample data")
  print(df[1, ])
}

# wait for key to continue
Pause = function() {
    cat ("Press [enter] to continue")
    line = readline()
}

## checks if a library is installed
IsInstalled = function(pkg) {
  is.element(pkg, installed.packages()[ , 1])
}

## it installs a library if its not installed already
InstallIf = function(pkg, F=InstallGooglePackages, ...) {
  if (!IsInstalled(pkg)) {
    F(pkg, ...)
  }
}

## it installs a library if not installed and it loads it regardless
InstallLoad = function(pkg) {
  InstallIf(pkg)
  library(pkg, character.only=TRUE)
}

## same as above but does it for set of libraries
InstallLoadBunch = function(pkgs) {
  for (pkg in pkgs) {
    InstallIf(
        pkg,
        #repos = c("http://rstudio.org/_packages", "http://cran.rstudio.com"),
        #dependencies=TRUE
    )

    eval(parse(text=paste0("library(package=", pkg, ")")))
  }
}

## purpose is to release memory by "removing" objects
# this function assigns NULL to the objects passed by their name in global env.
# then it collects the garbage twice (deliberately)
# setting to NULL is done since after "rm" and "gc"
# sometimes R does not release memory
Nullify = function(objectNames) {

  for (objectName in objectNames) {
    eval(parse(text=paste(objectName, "<<- NULL")))
    gc()
    gc()
  }
}

TestNullify = function() {

  x = rnorm(10^8)
  y = x + 1
  z = x + 2

  ## first nullify the objects
  Nullify(c("x", "y", "z"))

  ## then remove them if you like,
  # although they don't take space
  rm("x", "y", "z")

  ## Nullify seems to work inside a function as well
  x = rnorm(10^8)
  (function()Nullify("x"))()
  ## this will be true
  x == NULL

  ## Nullify seems to release memory after setting a local variable to NULL
  # as well
}

## Reading and Writing Files
# open the files in a dir; if fileList is not specific, we try to open all
OpenDataFiles = function(
    path, fileList=NULL, ReadF=read.csv,
    colClasses=NA, patterns=NULL,
    parallel=FALSE, parallel_outfile="") {

  if (is.null(fileList)) {
    fileList = list.files(path)
  }

  if (!is.null(patterns)) {
    fileListInd = do.call(
        intersect,
        lapply(X=patterns, FUN=function(p){grep(pattern=p, x=fileList)}))

    fileList = fileList[fileListInd]
  }

  k = length(fileList)
  if (k == 0) {
    warning("no files in path or no files followed the patterns specified.")
    return(NULL)
  }

  ## read one file
  F = function(i) {
    Src()
    fn0 = fileList[[i]]
    fn = paste(path, fn0, sep="")
    fn = file(fn)
    df = ReadF(fn, colClasses=colClasses)
    return(df)
  }


  if (!parallel) {
    dfList = lapply(X=1:k, FUN=F)
  } else {
    suppressMessages(library("parallel"))
    closeAllConnections()
    no_cores = detectCores() - 3
    no_cores = min(no_cores, k + 1)
    Mark(no_cores, "no_cores")
    # Initiate cluster
    cl = makeCluster(no_cores, outfile=parallel_outfile)
    clusterExport(
            cl=cl,
            list(
                "fileList", "path", "ReadF", "Src"),
            envir=environment())
    dfList =  parLapply(cl=cl, X=1:k, fun=F)
    stopCluster(cl)
    closeAllConnections()
  }

  names(dfList) = names(fileList)
  return(dfList)
}

## write a dataframe to a file with separators;
# can be used for qwiki for example.
WriteDataWithSep = function(
    fn, path=NULL, data, dataSep="|", headerSep="||") {

  options("scipen"=100, "digits"=4)
  fn = paste(path, fn, sep="")
  sink(fn)
  n = dim(data)[1]
  l = dim(data)[2]
  header = names(dat2)
  cat(headerSep)
  for (j in 1:l) {
    cat(header[j])
    cat(headerSep)
  }
  cat("\n")
  for (i in 1:n) {
    cat(dataSep)
    for (j in 1:l) {
      el = as.character(data[i, j])
      cat(el)
      cat(dataSep)
    }
    cat("\n")
  }
  sink()
}

## removes trailing spaces
TrimTrailing = function(x) {
  gsub("^\\s+|\\s+$", "", x)
}

## flatting a column with repeated field
# below we have another version of same function
# which is most likely faster for large data
Flatten_RepField_v1 = function(df, listCol, sep=NULL) {

  if (!is.null(sep)) {
    s = strsplit(as.character(df[ , listCol]), split=sep)
  }

  cols = colnames(df)
  cols2 = cols[cols != listCol]
  outDf = data.frame(listCol=unlist(s))
  colnames(outDf) = listCol
  for (col in cols2) {
    outDf[ , col] = rep(df[ , col], sapply(s, length))
  }
  return(outDf)
}

TestFlatten_RepField_v1 = function() {

  df = data.frame(list("var1"=c("a,b,c", "d,e,f"), "var2"=1:2, "var3"=3:4))
  print(df)
  Flatten_RepField_v1(df=df, listCol="var1", sep=",")

  df = data.frame(list("var1"=I(list(1:3, 4:6)), "var2"=1:2, "var3"=3:4))
  print(df)
  Flatten_RepField_v1(df=df, listCol="var1", sep=NULL)
}

## flatten a column (listCol) of df with multiple values
# the column elements could be lists
# or could be separated by sep e.g. comma
# this is faster than v1, but more complex
Flatten_RepField = function(df, listCol, sep=NULL) {

  if (!is.null(sep)) {
    F = function(x) {
      l = as.vector(strsplit(x, sep)[[1]])
      return(l)
    }

    df$newListCol = lapply(X=as.character(df[ , listCol]), FUN=F)
  } else {
    df$newListCol = df[ , listCol]
    df = DropCols(df=df, cols=listCol)
  }

  cols = names(df)[names(df) != "newListCol"]
  dt = data.table(df)[ , unlist(get("newListCol")), by=cols]
  df = data.frame(dt)
  df[ , listCol] = df[ , "V1"]
  df = df[ , ! names(df) %in% "V1"]
  return(df)
}

TestFlatten_RepField = function() {

  df = data.frame(list("var1"=c("a;b;c", "d;e;f"), "var2"=1:2, "var3"=3:4))
  print(df)
  Flatten_RepField(df=df, listCol="var1", sep=";")

  df = data.frame(list("var1"=I(list(1:3, 4:6)), "var2"=1:2, "var3"=3:4))
  print(df)
  Flatten_RepField(df=df, listCol="var1", sep=NULL)
}

## creating a single string column using multiple columns (cols)
# and adding that to the data frame
Concat_stringColsDf = function(df, cols, colName=NULL, sepStr="-") {

  x = ""
  if (is.null(colName)){
    colName = paste(cols, collapse=sepStr)
  }

  for (i in 1:length(cols)) {
    col = cols[i]
    x = paste0(x, as.character(df[ , col]))
    if (i < length(cols)) {
      x = paste0(x, "-")
    }
  }
  df[ , colName] = x
  return(df)
}

TestConcat_stringColsDf = function() {

  df = data.frame(list("a"=1:3, "b"=c("rr",  "gg", "gg"), "c"=1:3))
  Concat_stringColsDf(df=df, cols=c("a", "b", "c"), colName=NULL, sepStr="-")
}

## dropping some columns from a data frame
DropCols = function(df, cols) {
  return(df[ , !(names(df) %in% cols), drop=FALSE])
}

## subset columns of a data frame in a clear way
DfSubsetCols = function(df, keepCols=NULL, dropCols=NULL) {

    cols = colnames(df)
    if (!is.null(keepCols)) {
      cols = cols[cols %in% keepCols]
    }

    if (!is.null(dropCols)) {
      cols = cols[!(cols %in% dropCols)]
    }

    return(df[ , cols])
}

## subset columns of a data table in a clear way
DtSubsetCols = function(dt, keepCols=NULL, dropCols=NULL) {

    cols = colnames(dt)
    if (!is.null(keepCols)) {
      cols = cols[cols %in% keepCols]
    }

    if (!is.null(dropCols)) {
      cols = cols[!(cols %in% dropCols)]
    }

    return(dt[ , mget(cols)])
}

## simple aggregation with data.table
DtSimpleAgg = function(
    dt, gbCols=NULL, valueCols=NULL, cols=NULL, F=sum) {
  ## this aggregates multiple columns with the same function
  # this first subsets the data to the cols we need
  # then it aggregates with F

  # if we are not given the valueCols or all cols we need
  # we assume we need all cols in the dt
  if (is.null(cols) & (is.null(valueCols) | is.null(gbCols))) {
    cols = names(dt)
  }

  if (is.null(cols)) {
    cols = c(gbCols, valueCols)
  }

  if (is.null(gbCols)) {
    gbCols = setdiff(cols, valueCols)
  }

  outDt = dt[ , mget(cols)][ , lapply(.SD, F), by=gbCols]
  return(outDt)
}

## calculating bootstrap conf intervals for win/loss ratio
# input is a binary vector
# the idea is to use bootstrap
# for any bootstrap sample which degenerates
# because we have all ones or all zeros, we add (0,1) to the vector
# we also do that to the original vector!
# if we don't confidence intervals for vectors
# such as (0,0) will be one point (or (1,1))
Ci_forWLRatio = function(x) {

  flag = 'None'
  if (sum(x) == 0) {
    x = c(x, 1, 0)
    flag = 'Zero'
  }

  if (sum(1-x) == 0) {
    x = c(x, 0, 1)
    flag = 'Inf'
  }

  Bootst = function(data, F, num=1000) {

    n = dim(data)[1]
    G = function(i) {
      samp = sample(1:n, n, replace=TRUE)
      data2 = data[samp, , drop=FALSE]
      F(data2)
    }

    ind = as.list(1:num)
    res = lapply(X=ind, FUN=G)
    res =unlist(res)
    return(res)
  }

  data = data.frame(x)

  WL = function(data) {
    y = data[ , 1]
    if (sum(y) == 0) {y = c(y, 1, 0)}
    if (sum(1-y) == 0) {y = c(y, 0, 1)}
    out = sum(y) / sum(1-y)
    return(out)
  }

  res = Bootst(data=data, F=WL, num=1000)
  qVec = quantile(res, c(0.025, 0.975))

  ## we adjust the extremes of the interval in the degenerate case
  if (flag == 'Inf') {
    qVec[2] = Inf
  }

  if (flag == 'Zero') {
    qVec[1] = 0
  }

  return(qVec)
}

TestCi_forWLRatio = function() {

  Ci_forWLRatio(c(rep(1, 7)))
  Ci_forWLRatio(c(rep(1, 6), 0, 0, 0))
  Ci_forWLRatio(c(0, 0, 0))
  Ci_forWLRatio(c(1, 0, 0))
  Ci_forWLRatio(c(1, 1, 1))
}

## calculates CLT confidence interval
CltCi = function(x, p=0.95) {
  muHat = mean(x)
  error = qnorm(1 - (1-p)/2) * sd(x)/sqrt(length(x))
  upper = muHat + error
  lower = muHat - error
  return(list("muHat"=muHat, "error"=error, "upper"=upper, "lower"=lower))
}

## calculates CIs for multiple columns in a df and returns a df
CltCiDf = function(df, cols, p=0.95) {

  F = function(col) {
    x = df[ , get(col)]
    return(CltCi(x, p=p))
  }

  res = lapply(cols, FUN=F)
  names(res) = cols
  outDf = data.frame(matrix(unlist(res), nrow=4, byrow=TRUE))
  outDf[ , "metric"] = cols
  names(outDf) = c("muHat", "error", "upper", "lower", "metric")
  outDf = outDf[ , c("metric", "muHat", "error", "upper", "lower")]

  return(outDf)
}

## calculates relative risk
RelativeRiskCi = function(a1, n1, a2, n2) {

  p1 = a1/n1
  p2 = a2/n2
  if (p2 == 0) {
    print("the probability in the denom is zero, infinite risk!")
    return()
  }
  risk = p1/p2
  logRisk = log(risk)
  se = sqrt(1/a1 + 1/a2 - 1/n1 - 1/n2)
  logRiskUpper = logRisk + 1.96*se
  logRiskLower = logRisk - 1.96*se
  riskUpper = exp(logRiskUpper)
  riskLower = exp(logRiskLower)
  return(list(
      "risk"=risk,
      "riskLower"=riskLower,
      "riskUpper"=riskUpper,
      "logRisk"=logRisk,
      "logRiskUpper"=logRiskUpper,
      "logRiskLower"=logRiskLower,
      "logScaleError"=1.96*se
      ))
}

## calculates an upper bound/conservative CI for
# relative risk when the sample sizes are missing
# but their relative size is know
# e.g. this is true for experiment mods
RelativeRiskCi_approx = function(a1, a2, n2_n1_ratio=1) {

  risk = a1/a2 * n2_n1_ratio
  logRisk = log(risk)
  se = sqrt(1/a1 + 1/a2)
  logRiskUpper = logRisk + 1.96*se
  logRiskLower = logRisk - 1.96*se
  riskUpper = exp(logRiskUpper)
  riskLower = exp(logRiskLower)
  return(list(
      "risk"=risk,
      "riskLower"=riskLower,
      "riskUpper"=riskUpper,
      "logRisk"=logRisk,
      "logRiskUpper"=logRiskUpper,
      "logRiskLower"=logRiskLower,
      "logScaleError"=1.96*se
      ))
}

TestRelativeRiskCi_approx = function() {

  F = function(n1) {
    a1 = 30
    e1 = RelativeRiskCi(a1=a1, n1=n1, a2=2*a1, n2=3*n1)[["logScaleError"]]
    e2 = RelativeRiskCi_approx(a1=a1, a2=2*a1, n2_n1_ratio=3)[["logScaleError"]]
    return(c(e1, e2))
  }

  grid = (a1 + 1):200
  res = lapply(grid, FUN=F)
  outDf = data.frame(matrix(unlist(res), nrow=length(grid), byrow=TRUE))


  plot(
      grid, outDf[ , 1], ylim=c(0, 2*outDf[1, 2]),
      col="blue", ylab="CI error in log risk scale", xlab="n1")
  abline(h=outDf[1, 2], col="red")

  abline(v=2*a1, col="grey")
  text(x=2*a1, y=outDf[1, 2]/2, labels="2*a1")
  text(x=grid[length(grid)]-5, y=outDf[1, 2], labels="approx")
}

# remap low freq labels to a new label in data
# this is useful to avoid model breakage
# this also remaps NAs to the newLabel
# labelsNumMax decides whats the max number of labels allowed
Remap_lowFreqCategs = function(
    dt,
    cols,
    newLabels="other",
    otherLabelsToReMap=NULL,
    freqThresh=5,
    labelsNumMax=NULL) {

  if (!"data.table" %in% class(dt)) {
    warning("dt is not a data.table")
    return()
  }

  dt2 = copy(dt)

  k = length(cols)
  if (length(freqThresh) == 1) {
    freqThresh = rep(freqThresh, k)
  }

  if (length(newLabels) == 1) {
    newLabels = rep(newLabels, k)
  }

  if (!is.null(labelsNumMax) && labelsNumMax == 1) {
    labelsNumMax = rep(labelsNumMax, k)
  }


  GetFreqLabels = function(i) {
    col = cols[i]
    freqDt = data.table(table(dt2[ , get(col)]))
    colnames(freqDt) = c(col, "freq")
    freqDt = freqDt[order(freq, decreasing=TRUE)]
    freqLabels = freqDt[freq > freqThresh[i]][ , get(col)]
    if (!is.null(labelsNumMax)) {
      maxNum = min(length(freqLabels), labelsNumMax[i])
      freqLabels = freqLabels[1:maxNum]
    }

    if (!is.null(otherLabelsToReMap)) {
      freqLabels = setdiff(freqLabels, otherLabelsToReMap)
    }
    return(freqLabels)
  }

  freqLabelsList = lapply(X=1:k, FUN=GetFreqLabels)
  names(freqLabelsList) = cols

  F = function(dt) {
    for (i in 1:length(cols)) {
      col = cols[i]
      newLabel = newLabels[i]
      badLablesNum = sum(!dt[, get(col)] %in% freqLabelsList[[col]])
      if (badLablesNum > 0) {
        data.table::set(
            dt,
            i=which(!dt[ , get(col)] %in% freqLabelsList[[col]]),
            j=col,
            value=newLabel)}
    }
    return(dt)
  }

  return(list("dt"=F(dt2), "F"=F, "freqLabelsList"=freqLabelsList))
}

TestRemap_lowFreqCategs = function() {

  dt = data.table(data.frame(
      "country"=c("", rep("US", 10), rep("IN", 3), rep("FR", 10), "IR", ""),
      "gender"=c(
          "", rep("MALE", 10), rep("FEMALE", 10), rep("OTHER", 3), "NONE", ""),
      "value"=rnorm(26)))

  res = Remap_lowFreqCategs(
      dt=dt, cols=c("country", "gender"), otherLabelsToReMap=c(""),
      freqThresh=5)

  print(res)

  dt2 = data.table(data.frame(
      "country"=c("", rep("NZ", 10), rep("IN", 10), rep("FR", 3), "IR", ""),
      "gender"=c(
          "", rep("MALE", 10), rep("FEMALE", 10), rep("OTHER", 3), "NONE", ""),
      "value"=rnorm(26)))
  res[["F"]](dt2)
}

## quick check
CheckColFreqDt = function(dt, col) {

  freqDf = data.frame(table(as.character(dt[ , get(col)])))
  freqDf = freqDf[order(freqDf[ , "Freq"], decreasing=TRUE), ]
  rownames(freqDf) = NULL
  Mark(dim(freqDf), "dim(freqDf)")
  Mark(freqDf[1:min(50, nrow(freqDf)), ], "freqDf")
  return(freqDf)

}

## replaces all NAs in a data.table dt, for given cols
DtReplaceNa = function(dt, cols=NULL, replaceValue=0) {

  dt2 = copy(dt)
  if (is.null(cols)) {
    cols = names(dt2)
  }
  for (col in cols) {
    dt2[is.na(get(col)), (col) := replaceValue]
  }
  return(dt2)
}

## categorical mode
CategMode = function(x) {

  x = na.omit(x)
  if (length(x) == 0) {
    return(NULL)
  }
  ux = unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

## continuous mode
ContiMode = function(x) {

  d = density(x)
  d[["x"]][which.max(d[["y"]])]
}

## replacing NAs with mode or means
DtRemapNa = function(
    dt,
    cols=NULL,
    NumericReplace=function(x){mean(x, na.rm=TRUE)},
    FactorReplace=CategMode) {

  dt2 = copy(dt)

  if (is.null(cols)) {
    cols = names(dt2)
  }

  for (col in cols) {

    naProp = sum(is.na(dt2[ , get(col)])) / nrow(dt2)

    if (naProp > 0) {
      print(col)
      print(naProp)
      print(class(dt2[ , get(col)]))

      if (class(dt2[ , get(col)]) %in% c("numeric")) {
        set(dt2, i=which(is.na(dt2[ , get(col)])), j=col,
        value=mean(dt2[ , get(col)], na.rm=TRUE))
      }

      if (class(dt2[ , get(col)]) %in% c("integer")) {
        set(dt2, i=which(is.na(dt2[ , get(col)])), j=col,
        value=round(NumericReplace(dt2[ , get(col)])))
      }

      if (class(dt2[ , get(col)]) %in% c("factor", "character")) {
        set(dt2, i=which(is.na(dt2[ , get(col)])), j=col,
        value=CategMode(dt2[ , get(col)]))
      }
    }
  }

  return(dt2)
}

## remap values in a column (col) to other given values in newValues
DtRemap_colValues = function(dt, col, values, newValues, newCol=NULL) {

  if (length(values) != length(newValues)) {
    stop("Error: values and newValues do not have the same length")
  }

  if (is.null(newCol)) {
    newCol = col
  }

  F = function(x) {
    if (!(x %in% values)) {
      return(x)
    } else {
      i = which(values == x)[1]
      return(newValues[i])
    }
  }

  dt[ , newCol] = sapply(dt[ , get(col)], FUN=F)
  return(dt)
}

TestDtRemap_colValues = function() {

  dt = data.table(x=c("a", "b", "c", "a"), y=1:4)
  values = c("a", "b")
  newValues = c("A", "B")
  DtRemapColValues(dt, col="x", values=values, newValues=newValues)
}

## rounds the numeric columns in a df
RoundDf = function(df, num=1) {

  cols = names(df)[unlist(lapply(df, is.numeric))]
  if (is.null(cols)) {
    return(df)
  }
  df[ , cols] = round(df[ , cols], num)
  return(df)
}

## rounds the numeric columns in a dt
RoundDt = function(dt, num=1) {

  cols = names(dt)[which(sapply(dt, is.numeric))]
  if (is.null(cols)) {
    return(dt)
  }
  dt[ , cols] = round(dt[ , mget(cols)], num)
  return(dt)
}

## rounds the numeric columns using signif in a df
SignifDf = function(df, num=1) {

  cols = names(df)[unlist(lapply(df, is.numeric))]
  if (is.null(cols)) {
    return(df)
  }
  df[ , cols] = signif(df[ , cols], num)
  return(df)
}

## rounds the numeric columns using signif in a dt
SignifDt = function(dt, num=1) {

  cols = names(dt)[which(sapply(dt, is.numeric))]
  if (is.null(cols)) {
    return(dt)
  }
  dt[ , cols] = signif(dt[ , mget(cols)], num)
  return(dt)
}

## adding + or - to CIs to make it easy to find significant ones
StarCiDf = function(
    df, upperCol, lowerCol, upperThresh=0, lowerThresh=0,
    starCol="sig_stars") {

  df[ , starCol] = ""

  for (i in 1:length(upperThresh)) {
    ind = df[ , lowerCol] > upperThresh[i]
    if (length(ind) > 0) {
      df[ind, starCol] = paste0(df[ind, starCol], "+")
    }
  }

  for (i in 1:length(lowerThresh)) {
    ind = df[ , upperCol] < lowerThresh[i]
    if (length(ind) > 0) {
      df[ind, starCol] = paste0(df[ind, starCol], "-")
    }
  }

  return(df)
}

##
StarPvalueDf = function(
    df, pvalueCol="p-value", thresh=c(0.1, 0.05, 0.01, 0.001, 0.0001),
    starCol="pvalue_stars") {

  df[ , starCol] = ""

  ind = df[ , pvalueCol] < thresh[1]
    if (length(ind) > 0) {
      df[ind, starCol] = paste0(df[ind, starCol], ".")
  }

  for (i in 2:length(thresh)) {
    ind = df[ , pvalueCol] < thresh[i]
    if (length(ind) > 0) {
      df[ind, starCol] = paste0(df[ind, starCol], "*")
    }
  }

  return(df)
}

## this is the standard version of StarCiDf
TidyCiDf = function(
    df,  upperCol="ci_upper", lowerCol="ci_lower",
    upperThresh=c(1, 1.5, 2), lowerThresh=c(1, 0.75, 0.5), rounding=3) {

  df = StarCiDf(
      df=RoundDf(df, rounding), upperCol=upperCol, lowerCol=lowerCol,
      upperThresh=c(1, 1.5, 2), lowerThresh=c(1, 0.75, 0.5))

  return(df)
}

## Creates a table summary for the output of a regression model coefficients
# e.g. glm
RegMod_coefTableSumm = function(
    mod, label, dropVars="(Intercept)", keepVars=NULL, signif=2) {

  df = data.frame(summary(mod)[["coefficients"]])
  df = df[ , c("Estimate", "Std..Error", "Pr...t..")]
  colnames(df) = c("Estimate", "Sd", "p-value")
  df[ , "var"] = rownames(df)
  df = df[ , c("var", "Estimate", "Sd", "p-value")]
  df[ , "model_label"] = label
  if (!is.null(dropVars)) {
    df = df[!(df[ , "var"] %in% dropVars), ]
  }

  if (!is.null(keepVars)) {
    df = df[df[ , "var"] %in% keepVars, ]
  }

  df = SignifDf(df=df, num=signif)
  rownames(df) = NULL
  return(df)
}

## Creates a coef table summary for a list of models
RegModList_coefTableSumm = function(
    modList, labels=NULL, dropVars=NULL, keepVars=NULL, signif=2) {

  if (is.null(labels)) {
    labels = names(modList)
  }
  F = function(i) {
    mod = modList[[i]]
    label = labels[i]
    out = RegMod_coefTableSumm(
        mod=mod,
        label=label,
        dropVars=dropVars,
        keepVars=keepVars,
        signif=signif)
    return(out)
  }

  outDf = do.call(what=rbind, args=lapply(X=1:length(modList), FUN=F))
  return(outDf)
}

## xtable with vertical dividers
# we do not capitalize here as an exception
# since this is a minor tweak to existing function
xtable2 = function(x, caption="", label="label", ...) {

  MakeAlignString = function(x) {
    k = ncol(x)
    format_str = ifelse(sapply(x, is.numeric), "r", "l")
    out = paste0("|", paste0(c("r", format_str), collapse = "|"), "|")
    return(out)
  }

  return(xtable(x, caption=caption, label=label, ..., align=MakeAlignString(x)))
}

## entropy
Entropy = function(p) {
  if (min(p) < 0 || sum(p) <= 0) {
    pNorm = p[p > 0] / sum(p)
  }
  -sum(log2(pNorm)*pNorm)
}

## should work with both data frame and data table
SplitStrCol = function(df, col, sepStr) {

  dt = data.table(df)

  if (sepStr == "") {
    F = function(x) {
      nchar(x)
    }
    sepNums = nchar(as.character(dt[ , get(col)]))
    colNum = sepNums[1]
  } else {

    F = function(x) {
      lengths(regmatches(x, gregexpr(sepStr, x)))
    }
    sepNums = unlist(lapply(FUN=F, X=dt[ , get(col)]))
    # sepNums = dt[, F(get(col)), by = 1:nrow(dt)][ , V1]
    # second approach but it wasnt really faster
    colNum = sepNums[1] + 1
  }


  print(summary(sepNums))

  if (max(sepNums) != min(sepNums)) {
    warning("the strings do not have the same number of sepStr within.")
    return(NULL)
  }

  setDT(dt)[ , paste0(col, 1:colNum):=tstrsplit(
      get(col), sepStr, type.convert=TRUE, fixed=TRUE)]

  newCols = paste0(col, 1:colNum)
  return(list("dt"=copy(dt), "newCols"=newCols))
}

TestSplitStrCol = function() {

  df = data.frame(
    attr = c(1, 30 ,4 ,6 ),
    type = c('foo_and_bar_and_bar3', 'foo_and_bar_2_and_bar3')
  )

  sepStr = "_and_"
  col = "type"

  SplitStrCol(df=df, col=col, sepStr=sepStr)

  dt = data.table(df)
  SplitStrCol(df=dt, col=col, sepStr=sepStr)

  df = data.frame(
    attr = c(1, 30 ,4 ,6 ),
    type = c('aaa', 'abc')
  )

  SplitStrCol(df=data.table(df), col="type", sepStr="")
}

## this applies the jack-knife method
# F is an estimator which is a function of dt and returns a vector
# we want a CI for each of the components returned by F
# dt is a data.table
PartitionCi = function(
    dt, Estim, bucketCol=NULL, bucketNum=NULL, method="jk", conf=0.95) {

  if (is.null(bucketCol)) {
    bucketCol = "bucket"
    n = nrow(dt)
    bucketSize = floor(n / bucketNum)
    r = n - bucketSize*bucketNum
    bucketVec = c(rep(1:bucketNum, bucketSize))
    if (r > 0) {
      bucketVec = c(bucketVec, 1:r)
    }
    bucketVec = sample(bucketVec)
    dt[ , "bucket"] =  bucketVec
  }

  buckets = unique(dt[ , get(bucketCol)])

  Jk = function(b) {
    dt2 = dt[get(bucketCol) != b]
    dt2 = DtSubsetCols(dt2, dropCols=bucketCol)
    return(Estim(dt2))
  }

  Simple = function(b) {
    dt2 = dt[get(bucketCol) == b]
    dt2 = DtSubsetCols(dt2, dropCols=bucketCol)
    return(Estim(dt2))
  }

  if (method == "jk") {
    G = Jk
  } else {
    G = Simple
  }

  estimList = lapply(X=buckets, FUN=G)
  x0 = estimList[[1]]
  names = names(x0)

  estimDf = setNames(
      data.frame(matrix(ncol=length(x0), nrow=length(buckets))),
      names)

  for (i in 1:length(buckets)) {
    estimDf[i, ] = estimList[[i]]
  }

  CltCi = function(x) {
    x = na.omit(x)
    m = mean(x)
    s = sd(x)
    n = length(x)

    if (method == "jk") {
      estimSd = sqrt(n-1) * s
    } else {
      estimSd = s / sqrt(n)
    }

    zValue = 1 - (1-conf) / 2
    upper = m + qnorm(zValue) * estimSd
    lower = m - qnorm(zValue) * estimSd
    return(c(m, estimSd, lower, upper))
  }

  ciDf = t(apply(estimDf, 2, CltCi))
  ciDf = data.frame(ciDf)
  colnames(ciDf) = c("mean", "estim sd", "lower", "upper")
  ciDf[ , "length"] = ciDf[ , "upper"] - ciDf[ , "lower"]
  return(ciDf)
}

BootstrapCi = function(dt, Estim, bsNum=500, conf=0.95) {

  q1 = (1 - conf) / 2
  q2 = 1 - q1

  n = nrow(dt)
  Bs = function(b) {
    samp = sample(1:n, replace=TRUE)
    dt2 = dt[samp, ]
    return(Estim(dt2))
  }


  estimList = lapply(X=1:bsNum, FUN=Bs)
  x0 = estimList[[1]]
  names = names(x0)

  estimDf = setNames(
      data.frame(matrix(ncol=length(x0), nrow=bsNum)),
      names)

  for (i in 1:bsNum) {
    estimDf[i, ] = estimList[[i]]
  }

  BsCi = function(x) {
    x = na.omit(x)
    m = mean(x)
    estimSd = sd(x)
    upper = quantile(x, q2)
    lower = quantile(x, q1)
    return(c(m, estimSd, lower, upper))
  }

  ciDf = t(apply(estimDf, 2, BsCi))
  ciDf = data.frame(ciDf)
  colnames(ciDf) = c("mean", "estim sd", "lower", "upper")
  ciDf[ , "length"] = ciDf[ , "upper"] - ciDf[ , "lower"]
  return(ciDf)
}

TestPartitionCi = function() {

  n = 10^4
  x1 = rnorm(n, mean=3, sd=10)
  x2 = rnorm(n, mean=5, sd=2)
  x3 = x1 + x2
  df = data.frame(x1=x1, x2=x2, x3=x3)
  dt = data.table(df)
  bucketCol = NULL
  bucketNum = 20

  Estim = function(dt) {colMeans(dt)}
  PartitionCi(
      dt=dt, Estim=Estim, bucketCol=NULL, bucketNum=bucketNum,
      method="jk", conf=0.95)

  PartitionCi(
      dt=dt, Estim=Estim, bucketCol=NULL, bucketNum=bucketNum,
      method="simple", conf=0.95)

  BootstrapCi(dt=dt, Estim=Estim, bsNum=1000, conf=0.95)

  Estim = function(dt) {mean(dt[[1]])}
  PartitionCi(
      dt=dt, Estim=Estim, bucketCol=NULL, bucketNum=bucketNum,
      method="jk", conf=0.95)

  PartitionCi(
      dt=dt, Estim=Estim, bucketCol=NULL, bucketNum=bucketNum,
      method="simple", conf=0.95)

  BootstrapCi(dt=dt, Estim=Estim, bsNum=500, conf=0.95)
}

## substituting multiple values
ReplaceStringMulti = function(x, values, subs) {

  for (i in 1:length(values)) {
    x = gsub(values[i], subs[i], x)
  }

  return(x)
}

## capitalizes all words in a sentence
CapWords = function(x, splitStr=" ") {
  s = strsplit(x, splitStr)[[1]]

  paste(toupper(substring(s, 1, 1)), substring(s, 2),
      sep="", collapse=splitStr)
}

TestCapWords = function() {

  CapWords("be free.") == "Be Free."
}

# Cartesian product of string vectors
StringCartesianProd = function(..., prefix="", sep="_") {

  #paste0(prefix, levels(interaction(..., sep=sep)))
  paste2 = function(...) {
    paste(..., sep=sep)
  }

  df = expand.grid(...)
  do.call(what=paste2, args=df)
}

## test for the above function
TestStringCartesianProd = function() {

  values = c("active_days_num", "activity_num")
  products = c("assist", "search", "watchFeat", "photos", "multi")
  periods = c("pre", "post")
  valueCols = StringCartesianProd(values, products, periods, sep="_")
}

## sorts data frames and data.tables
# R syntax for sorting is ineffective and not so great inside functions
# this function provides a user friendly approach
# cols: columns to be used for sorting, in order of their importance
# ascend: specifies if the order is ascending (TRUE) or not (FALSE)
# default for ascend is (TRUE, ..., TRUE)
SortDf = function(
    df, cols=NULL, ascend=NULL, printCommand=FALSE) {

  if (is.null(cols)) {
    cols = names(df)
  }

  if (min(cols %in% names(df)) < 1) {
    warning("some of your columns are not in df.")
    return(df)
  }

  if (is.null(ascend)) {
    ascend = rep(TRUE, length(cols))
  }

  commandStr = "order("

  for (i in 1:length(cols)) {
    dir = ascend[i]
    col = cols[i]
    if (dir) {
      commandStr = paste0(commandStr, " ", col)
    } else {
      commandStr = paste0(commandStr, " ", "-", col)
    }

    if (i == length(cols)) {
      commandStr = paste0(commandStr, ")")
    } else {
      commandStr = paste0(commandStr, ",")
    }
  }

  commandStr = paste0("df = df[with(df, ", commandStr,") , ]")

  if (printCommand) {
    print(commandStr)
  }

  eval(parse(text=commandStr))

  return(df)
}

TestSortDf = function() {

  n = 20

  df = data.frame(
      "first_name"=sample(c("John", "Omar", "Mo"), size=n, replace=TRUE),
      "family_name"=sample(c("Taylor", "Khayyam", "Asb"), size=n, replace=TRUE),
      "grade"=sample(1:10, size=n, replace=TRUE))

  # sort with defaults, nice and easy
  SortDf(df=df)

  # choose columns and the direction of sorting
  SortDf(
      df=df,
      cols=c("first_name", "family_name", "grade"),
      ascend=c(TRUE, TRUE, FALSE),
      printCommand=TRUE)

  # try same with data table object
  SortDf(
      df=data.table(df),
      cols=c("first_name", "family_name", "grade"),
      ascend=c(TRUE, TRUE, FALSE),
      printCommand=FALSE)
}

## this returns a function which calculates a relative err
# using norms
# this error function is symmetric with respect to its inputs position
# p denoted the power in L-p norm (p > 0)
SymRelErrFcn = function(p) {

  if (p <= 0) {
    warning("p has to be positive")
    return(NULL)
  }

  F = function(x, y) {
    z = abs(x - y)
    err = 2 * z^p / (abs(x)^p + abs(y)^p)
    return(err)
  }

  return(F)
}

TestSymRelErrFcn = function() {
  x = 3
  y = 5

  SymRelErrFcn(2)(x, y)
  SymRelErrFcn(1)(x, y)
}

## calculates diff between valueCols between two data frames
CalcErrDfPair = function(
    df1, df2, valueCols, Err, ErrAvgF=mean, sort=TRUE,
    idCols=NULL, checkMatch=TRUE) {

  if (nrow(df1) != nrow(df2)) {
    warning("length of the data frames is not the same.")
    return(NULL)
  }

  if (is.null(idCols)) {
    ## the common cols except for valueCols used for sorting and matching
    idCols = setdiff(intersect(colnames(df1), colnames(df2)), valueCols)
  }

  if (sort) {
    df1 = SortDf(df=df1, cols=idCols)
    df2 = SortDf(df=df2, cols=idCols)
  }

  if (checkMatch) {
    if (!identical(df1[ , idCols], df2[ , idCols])) {
      warning(paste(
          "id columns:",
          paste(idCols, collapse=" "),
          "are not matching in values."))

      return(NULL)
    }
  }

  errVec = NULL

  for (valueCol in valueCols) {
    err = ErrAvgF(Err(df1[ , valueCol], df2[ , valueCol]))
    errVec = c(errVec, err)
  }

  names(errVec) = valueCols

  return(errVec)
}

TestCalcErrDfPair = function() {

  n = 100
  x1 = sample(1:n, size=n)
  x2 = sample(1:n, size=n)

  df1 = data.frame(
      "x1"=x1,
      "x2"=x2,
      "y1"=2*x1 + rnorm(n),
      "y2"=2*x2 + rnorm(n))

  df2 = data.frame(
      "x1"=x1,
      "x2"=x2,
      "y1"=2*x1 + rnorm(n),
      "y2"=1*x2 + rnorm(n))

  df3 = df2[sample(1:n, n), ]

  Err = SymRelErrFcn(2)

  CalcErrDfPair(
      df1=df1, df2=df2, valueCols=c("y1", "y2"),
      Err=Err, ErrAvgF=mean, sort=TRUE,
      idCols=NULL, checkMatch=TRUE)

  CalcErrDfPair(
      df1, df3, valueCol=c("y1", "y2"), Err=Err, ErrAvgF=mean, sort=TRUE,
      idCols=NULL, checkMatch=TRUE)
}

## This function compares two frequency tables
# it returns a row_wise err which is then averaged across rows
# also returns a global err which is standardized by the total freq
# both metrics are symmetric
# note that this is not a distbn distance by default: set distbn_dist=TRUE
# this will compare frequencies by default
FreqTables_simpleDiff = function(
    tab1, tab2, AvgF=mean, distbn_dist=FALSE) {

  df1 = data.frame(tab1)
  colnames(df1) = c("var", "freq1")

  df2 = data.frame(tab2)
  colnames(df2) = c("var", "freq2")

  ## if we want a distbn distance we cal probabilities
  if (distbn_dist) {
    df1[ , "freq"] = df1[ , "freq"] / sum(df1[ , "freq"])
    df2[ , "freq"] = df2[ , "freq"] / sum(df2[ , "freq"])
  }

  compareDf = merge(df1, df2, on=colnames(df1), all=TRUE)
  compareDf[is.na(compareDf)] = 0
  compareDf[ , "err"] = abs(compareDf[ , "freq1"] - compareDf[ , "freq2"])

  denom_elementwise = (0.5*compareDf[ , "freq1"] + 0.5*compareDf[ , "freq2"])
  avg_elementwise_err = AvgF(compareDf[ , "err"] / denom_elementwise)

  total_freq = sum(compareDf[ , "freq1"]) + sum(compareDf[ , "freq2"])
  global_err = sum(compareDf[ , "err"]) / total_freq

  return(list(
      "avg_elementwise_err"=avg_elementwise_err,
      "global_err"=global_err))
}

## which value is the min
MinInd = function(x) {
  which(x == min(x))
}

# which row has the min value for col
MinIndDf = function(df, col) {

  x = df[ , col]
  ind = which(x == min(x))
  return(df[ind, , drop=FALSE])
}

## debugging R code
Example = function() {

  f = function() {
    on.exit(traceback(1))
    g = function() {
      x = 1 + "a"
    }
    g()
  }
  f()

  #traceback()
}

## for debugging within R
Debug = function(F)  {
    on.exit(traceback(1))
    F()
    #traceback()
}

## check for a library dependencies
# also tries to find out if those libs are installed by checking library(lib)
# if not installed, it tries to install them
# it reports un-installed ones and the unavailable ones for install
# Install is either install.packages or a custom install function
Check_andFix_dependencies = function(lib, Install) {

  library("tools")
  libs = package_dependencies(lib)[[1]]
  uninstalledLibs = NULL
  unavailLibs = NULL

  F = function(lib) {
    suppressMessages(library(lib, character.only=TRUE))
    return(NULL)
  }

  for (lib in libs) {

    x = tryCatch(
        F(lib),
        error=function(e) {lib})
    uninstalledLibs = c(uninstalledLibs, x)

  }

  F = function(lib) {
    suppressMessages(Install(lib))
    return(NULL)
  }

  for (lib in uninstalledLibs) {

    x = tryCatch(
        F(lib),
        error=function(e) {lib})
    unavailLibs = c(unavailLibs, x)
  }

  return(list(
      unavailLibs=unavailLibs,
      uninstalledLibs=uninstalledLibs))
}
