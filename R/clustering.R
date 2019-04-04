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

## functions for creating clusters using K-means and plotting them

## subset/slice df according to a condition list.
SliceDfCondition = function(df, conditions) {

  conditionElement = conditions[["elem"]]
  conditionBigger = conditions[["bigger"]]
  conditionSmaller = conditions[["smaller"]]
  conditionFcns = conditions[["fcns"]]

  sliceName = ""

  if (!is.null(conditionElement)) {
    cols = names(conditionElement)
    for (i in 1:length(cols)) {
      col = cols[i]
      set = conditionElement[[col]]
      setName = paste(as.character(set), collapse=",", sep="")
      condName = paste(col, "=", setName, ", ", collapse="", sep="")
      sliceName = paste(sliceName, condName, collapse="&", sep="")
      df = df[df[ , col] %in% set, ]
    }
  }

  if (!is.null(conditionBigger)) {
    cols = names(conditionBigger)
    for (i in 1:length(cols)) {
      col = cols[i]
      value = conditionBigger[[col]]
      df = df[df[ , col] > value, ]
      valueName = as.character(value)
      condName =  paste(col, ">", valueName, ", ", collapse="", sep="")
      sliceName = paste(sliceName, condName, collapse="", sep="")
    }
  }

  if (!is.null(conditionSmaller)) {
    cols = names(conditionSmaller)
    for (i in 1:length(cols)) {
      col = cols[i]
      value = conditionSmaller[[col]]
      df = df[df[ , col] < value, ]
      valueName = as.character(value)
      condName =  paste(col, "<", valueName, ", ", collapse="", sep="")
      sliceName = paste(sliceName, condName, collapse="", sep="")
    }
  }

  if (!is.null(conditionFcns)) {
    cols = names(conditionFcns)
    for (i in 1:length(cols)) {
      col = cols[i]
      Fcn = conditionFcns[[col]]
      df = df[Fcn(df[ , col]), ]
    }
  }

  return(list("df"=df, "sliceName"=sliceName))
}

TestSliceDfCondition = function() {

  df0 = data.frame("city"=c("a", "b", "c", "d", "e"), "pop"=c(1, 5, 6, 8, 1))
  print(SliceDfCondition(df0, conditions=list("elem"=list("city"=c("a", "b")))))
  print(SliceDfCondition(df0, conditions=list("bigger"=list("pop"=c(4.5)))))
  print(
      SliceDfCondition(
          df0,
          conditions=list(
              "elem"=list("city"=c("a", "b", "c")),
              "bigger"=list("pop"=c(4.5)))))
}

# Builds a frequency table for x given cutoffs
BuildFreqTable = function(x, cutoffs=NULL, quantileL=0.2, prop=FALSE) {

  l = quantileL
  if (is.null(cutoffs)) {
    cutoffs = c(-Inf, quantile(x, seq(l, 1-l, l), na.rm=TRUE), +Inf)
    cutoffs = unique(cutoffs)
  }

  z = cut(x, cutoffs)
  tab = table(z)
  if (prop) {
    tab = 100 * tab / sum(tab)
  }
  return(tab)
}

TestBuildFreqTable = function() {
  y = 1:100
  BuildFreqTable(y, quantileL=0.3)
}

## build frequency tables for multiple columns (cols) in df
## it uses the same cutoffs
## it returns a table data frame
## it also returns a table for the total (all data combined)
BuildFreqTableDf = function(
    df, cols=NULL, cutoffs=NULL, quantileL=0.2, prop=FALSE) {

  l = quantileL
  if (is.null(cols)) {
    cols = colnames(df)
  }

  df = df[ , cols, drop=FALSE]

  if (is.null(cutoffs)) {
    cutoffs = c(-Inf, quantile(df, seq(l, 1-l, l), na.rm=TRUE), +Inf)
    cutoffs = unique(cutoffs)
  }

  x = as.vector(as.matrix(df))
  tab = BuildFreqTable(x, cutoffs=cutoffs, quantileL=NULL)
  if (prop) {
    tab = 100 * tab/sum(tab)
  }

  outDf = data.frame(tab)
  colnames(outDf) = c("partition", "total")

  for (i in 1:length(cols)) {
    x = df[ , cols[i]]
    tab = BuildFreqTable(x, cutoffs=cutoffs, quantileL=NULL)
    if (prop) {
      tab = 100 * tab/sum(tab)
    }
    outDf0 = data.frame(tab)
    colnames(outDf0) = c("partition", cols[i])
    outDf = merge(outDf, outDf0, how="left", sort=FALSE)
  }

  return(outDf)
}

TestBuildFreqTableDf = function() {
  df0 = data.frame("x"=1:10, "y"=2:11)
  tabDf = BuildFreqTableDf(df0)
  Plt_barsSideBySide(tabDf, cols=c("total", "x", "y"), legendPosition="topleft")
}

## this build tab Df and then plots it SxS
BuildFreqTable_pltSideBySide = function(
    df, cols=NULL, cutoffs=NULL, quantileL=0.15, prop=FALSE,
    legendPosition="topright",  xlim=NULL, ylim=NULL,
    xlab=NULL, ylab=NULL, legendLabels=NULL, colors=NULL, xpd=FALSE) {

  tabDf = BuildFreqTableDf(
      df, cols=cols, cutoffs=cutoffs, quantileL=quantileL, prop=prop)

  allCols = colnames(tabDf)

  if (is.null(legendLabels)) {
    xLabels = tabDf[ , 1]
  }
  cols = allCols[-(1:2)]

  Plt_barsSideBySide(
      df=tabDf, cols=cols, legendPosition=legendPosition,
      xLabels=xLabels, xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab,
      legendLabels=legendLabels, colors=colors, xpd=xpd)

  return(tabDf)
}


## add a dot between digits and letters, if a string  has both
SepDigits = function(x, sep=".") {

  digits = gsub("[^[:digit:]]", "", x)
  letters = gsub("[0-9]", "", x)
  ind = which(digits != "")
  out = letters

  if (length(ind) > 0) {
    out[ind] = paste(letters[ind], digits[ind], sep=sep)
  }
  return(out)
}

# a function to replace NaNs similar to is.na which is an existing method
IsNanDf = function(x) {
  do.call(cbind, lapply(x, is.nan))
}

## add an avgCol to a df
AddAvgCol = function(
    df, respCols, idCols=NULL, avgColName="avgEng", AvgFcn=rowMeans) {

  if (is.null(idCols)) {
    idCols =  setdiff(colnames(df), respCols)
  }

  outDf = df[ , respCols, drop=FALSE]
  outDf[ , avgColName] = AvgFcn(outDf)

  if (length(idCols) > 0) {
    outDf = cbind(df[ , idCols, drop=FALSE], outDf)
  }

  return(outDf)
}

# this builds a prop data set
BuildPropDf = function(
    df, idCols, respCols, avgColName="avgEng", replaceNan=TRUE) {

  df2 = df[ , respCols, drop=FALSE]
  propDf =  prop.table(as.matrix(df2), margin = 1)
  propDf = cbind(df[ , idCols, drop=FALSE], propDf)
  propDf[ , avgColName] = rowMeans(df2)
  if (replaceNan) {
    propDf[is.nan(propDf)] = 0
  }
  return(propDf)
}

# this creates a prop Df using a list of subsets of columns,
# for each subset it calculates
BuildPropDfMulti = function(
    df, idCols, respColsList, avgColName="avgEng", replaceNan=TRUE) {

  n = length(respColsList)

  outDf = NULL
  for (i in 1:n) {
    respCols = respColsList[[i]]
    avgColName0 = paste0(avgColName, i)
    propDf0 = BuildPropDf(
        df=df[ , c(idCols, respCols), drop=FALSE], idCols=idCols,
        respCols=respCols, avgColName=avgColName0, replaceNan=replaceNan)
    if (i == 1) {
      outDf = propDf0
    } else {
      outDf = cbind(outDf, propDf0[ , c(respCols, avgColName0), drop=FALSE])
    }
  }
  return(outDf)
}

## this builds a diff df using a pattern such as col1, col2
## it detects col pairs according to the pattern and then calculates delta
BuildDiffDfSuffix = function(df, colPrefix=NULL, colSuffix=1:2, idCols=NULL) {

  if (is.null(colPrefix)) {
      respCols = colnames(df)[grepl("\\d", colnames(df))]
      ## omit the digits from the resp col names
      colPrefix = unique(gsub("[0-9]", "", respCols))
      colPrefix = gsub("\\.", "", colPrefix)
  }

  if (is.null(idCols)) {
    respCols = colnames(df)[grepl("\\d", colnames(df))]
    idCols = setdiff(colnames(df), respCols)
  }

  diffDf = df[ , idCols, drop=FALSE]

  for (i in 1:length(colPrefix)) {
    col = colPrefix[i]

    diffDf[ , paste0("d_", col)] = (
        df[ , paste0(col, colSuffix[2])] -
        df[ , paste0(col, colSuffix[1])])
  }

  diffCols = paste0("d_", colPrefix)

  return(list("df"=diffDf, "cols"=diffCols))
}

## calc pca
CalcPca = function(df, cols=NULL, scale=FALSE) {

  if (!is.null(cols)) {
    df = df[ , cols, drop=FALSE]
  }

  df = df[complete.cases(df), , drop=FALSE]

  # apply PCA - scale. = TRUE is highly
  # advisable, but default is FALSE.
  pca = prcomp(df, center = TRUE, scale. = scale)
  return(pca)
}

## assign middles for a set of centers
AddMiddleCl = function(dfCl, df) {

  dfCl[["middles"]] = dfCl[["centers"]]

  for (i in 1:length(dfCl[["size"]])) {
    ind = which(dfCl[["cluster"]] == i)

    if (length(ind) == 1) {
       dfCl[["middles"]][i, ] = as.numeric(df[ind, ])
    } else {
      df2 = df[ind, , drop=FALSE]
      dist = rowSums(sapply(df2, function(x) x - mean(x))^2)
      ind2 = which(dist == min(dist))[1]
      dfCl[["middles"]][i, ] = as.numeric(df2[ind2, ])
    }
  }
  return(dfCl)
}

## kmeans alg for data frames
## this function provides cluster middles rather than cluster centers only
KmeansDf = function(
    df, centers, cols=NULL, iter.max=15, nstart=nstart, ss=NULL) {
  #set.seed(20)
  if (!is.null(ss) && dim(df)[1] > ss) {
    ind = sample(1:dim(df)[1], ss)
    df = df[ind, , drop=FALSE]
  }

  if (!is.null(cols)) {
    df = df[ , cols, drop=FALSE]
  }

  df = df[complete.cases(df), , drop=FALSE]

  dfCl = kmeans(df, centers=centers, iter.max=iter.max, nstart=nstart)
  dfCl = AddMiddleCl(dfCl=dfCl, df=df)

  return(dfCl)
}

## this buckets the data
BucketDf = function(df, bucketSize=NULL, bucketNum=NULL, cols=NULL) {

  if (!is.null(cols)) {
    df = df[ , cols, drop=FALSE]
  }

  df = df[complete.cases(df), , drop=FALSE]

  if (is.null(bucketSize)) {
    bucketSize = round(dim(df)[1] / bucketNum)
    bucketSize = max(bucketSize, 1)
  }

  bucketNum = dim(df)[1] %/% bucketSize
  if (bucketNum <= 1) {
    print("WARNING: bucket size is less than or equal to 1")
  }

  ss = bucketNum*bucketSize
  df = df[1:ss, , drop=FALSE]
  n = bucketSize
  dfBu = list()
  cluster = list(rep(1:(nrow(df)%/%n+1), each=n, len=nrow(df)))
  dfBu[["cluster"]] = unlist(cluster)
  centers = aggregate(df, cluster, mean)[-1]
  dfBu[["centers"]] = centers
  dfBu[["size"]] = rep(bucketSize, bucketNum)
  Fcn = function(x){sum((x-mean(x))^2)}
  ssDf = aggregate(df, list(rep(1:(nrow(df)%/%n+1), each=n, len=nrow(df))), Fcn)[-1]
  dfBu[["withinss"]] = rowSums(ssDf)
  dfBu = AddMiddleCl(dfCl=dfBu, df=df)
  return(dfBu)
}

Example = function() {
  m = 10
  k = 3
  set.seed(2)
  df = matrix(sample(1:(m*k), m*k), m, k)
  df = data.frame(df)
  n = 3
  list(rep(1:(nrow(df)%/%n+1), each=n, len=nrow(df)))
  F = function(x) {
    sum((x-mean(x))^2)}
  meanDf = aggregate(
      df, list(rep(1:(nrow(df)%/%n+1), each=n, len=nrow(df))), mean)[-1]
  ssDf = aggregate(
      df, list(rep(1:(nrow(df)%/%n+1), each=n, len=nrow(df))), Fcn)[-1]
  print(df)
  print("*** mean")
  print(meanDf[1:2, ])
  print("*** ss")
  print(ssDf[1:2, ])
  F(c(2, 15, 23))
}

## assess clustering
AssessClust = function(dfCl, plotIt=FALSE) {

  names(dfCl)
  clustNum = dim(dfCl[["centers"]])[1]
  dataDim =  dim(dfCl[["centers"]])[2]
  sizesSum = sum(dfCl[["size"]])
  withinPerClust = sqrt(dfCl[["withinss"]] / (dfCl[["size"]] * dataDim))
  avgWithin = sqrt(sum(dfCl[["withinss"]]) / ((sizesSum-clustNum) * dataDim))

  assesspltList = list()
  assesspltList[[1]] = qplot(
      x=1:clustNum, y=0, xend=1:clustNum, yend=withinPerClust,
      geom="segment", color=I(ColAlpha("blue", 0.5)), xlab="clust/bucket",
      ylab="withinPerClust", size=I(20)) +
      theme(legend.position="none")
  assesspltList[[2]] = qplot(
      x=1:clustNum, y=0, xend=1:clustNum, yend=dfCl[["withinss"]],
      geom="segment", color=I(ColAlpha("blue", 0.5)), xlab="clust/bucket",
      ylab="withinss", size=I(20)) +
      theme(legend.position="none")
  assesspltList[[3]] = qplot(
      x=1:clustNum, y=0, xend=1:clustNum, yend=dfCl[["size"]],
      geom="segment", color=I(ColAlpha("blue", 0.5)), xlab="clust/bucket",
      ylab="size (%)", size=I(20)) +
      theme(legend.position="none")

  if (plotIt) {
    print(assesspltList)
  }

  #avgBetween = sqrt(dfCl[["betweenss"]] / (dataDim * clustNum))
  #withinOnBetween = avgWithin / avgBetween
  withinOnTotal = dfCl[["tot.withinss"]] / dfCl[["totss"]]
  avgTotal = sqrt(dfCl[["totss"]]/(sizesSum * dataDim))

  dfCl[["withinPerClust"]] = withinPerClust
  dfCl[["avgWithin"]] = avgWithin
  dfCl[["avgTotal"]] = avgTotal
  dfCl[["withinOnTotal"]] = withinOnTotal
  dfCl[["assesspltList"]] = assesspltList

  return(dfCl)
}

## compare PCAs
ComparePca = function(
    pca1, pca2, pcaNum=4, xlim=c(-2, 2), ylim=c(-2, 2), pltTitle="",
    fontSizeAlpha=1) {

  pca_raw = pca1[["rotation"]]
  pca_cl = pca2[["rotation"]]

  pcapltList = list()

  for (i in 1:min(pcaNum, dim(pca_cl)[2])) {
    pcapltList[[i]] = qplot(x=pca_raw[ , i], y=pca_cl[ , i],
      ylab=paste("pca clust: comp", i), xlab=paste("pca raw: comp", i),
      color=ColAlpha("blue", 0.5)) + theme(legend.position="none") +
      geom_abline(intercept=0, slope=1, colour=ColAlpha("grey", 0.5)) +
      geom_abline(intercept=0, slope=-1, colour=ColAlpha("grey", 0.5)) +
      xlim(xlim) + ylim(ylim) + ggtitle(pltTitle) +
      theme(
          plot.title=element_text(face="bold", size=16*fontSizeAlpha, hjust=0))
  }

  return(pcapltList)
}

# this compare continuous multi distributions in terms of corr and pca
CompareContMultiDist = function(
    df1, df2, cols=NULL, mainText=c("", ""), fontSizeAlpha=1, pcaPltTitle="") {

  if (!is.null(cols)) {
    df1 = df1[ , cols, drop=FALSE]
    df2 = df2[ , cols, drop=FALSE]
  }

  corpltList = list()
  corpltList[["cor1"]] = CorPlt(df=df1, cols=NULL, colRange=c("yellow", "blue"),
    mainText=mainText[1], fontSizeAlpha=fontSizeAlpha)
  corpltList[["cor2"]] = NULL

  if (dim(df2)[1] > 1) {
    corpltList[["cor2"]] = CorPlt(df=df2, cols=NULL, colRange=c("yellow", "blue"),
      mainText=mainText[2], fontSizeAlpha=fontSizeAlpha)
  }

  pca1 = CalcPca(df=df1, cols=NULL)
  pca2 = NULL
  pcapltList = NULL

  if (dim(df2)[1] > 1) {
    pca2 = CalcPca(df=df2, cols=NULL)
    pcapltList = ComparePca(
        pca1, pca2, pltTitle=paste(pcaPltTitle, "PCA vs raw data PCA"),
        fontSizeAlpha=fontSizeAlpha)
  } else {
    print("WARNING: only one row in  df2 so we could not caculate PCA")
  }

  outList = list()
  outList[["pca1"]] = pca1
  outList[["pca2"]] = pca2
  outList[["corpltList"]] = corpltList
  outList[["pcapltList"]] = pcapltList

  return(outList)
}

## it clusters using kmeans and adds more assessment metrics
ClustAndAssess = function(
    df, centers, method="kmeans", cols=NULL, nstart=20, iter.max=15, ss=NULL,
    plotIt=FALSE, fontSizeAlpha=1, pltTitleSuff="") {

  if (!is.null(cols)) {
    df = df[ , cols, drop=FALSE]
  }

  if (method == "bucket") {
    dfCl = BucketDf(df=df, bucketSize=NULL, bucketNum=centers, cols=cols)
  } else if (method == "kmeans") {
    dfCl = KmeansDf(
        df=df, centers=centers, cols=cols, nstart=nstart, iter.max=iter.max,
        ss=ss)
  } else {
    print("WARNING: you specified a method which is not implemented")
    return(NULL)
  }

  dfCl = AssessClust(dfCl)
  outList = CompareContMultiDist(
      df1=df, df2=data.frame(dfCl[["centers"]]),
      cols=NULL, mainText=c(paste0("Raw", pltTitleSuff),
      paste0("Clusters: ", method, pltTitleSuff)),
      pcaPltTitle=paste0(method, pltTitleSuff), fontSizeAlpha=fontSizeAlpha)
  dfCl[["pcaRaw"]] = outList[["pca1"]]
  dfCl[["pcaClust"]] = outList[["pca2"]]
  dfCl[["pcapltList"]] = outList[["pcapltList"]]
  dfCl[["corpltList"]] = outList[["corpltList"]]

  if (plotIt) {
    print(dfCl[["corpltList"]])
    print(dfCl[["pcapltList"]])
  }

  return(dfCl)
}


TestClustAndAssess = function() {
  #source(src)
  # variable numbers
  k = 4
  #
  m = 50
  l = matrix(0, k, k)
  diag(l) = 1
  l[2, ] = c(5, 1, 0, 0 )
  l[3, ] = c(-3, -4, 1,0)
  l[4, ] = c(-3, -1, 2,1)
  sig = t(l) %*% l
  #print(l)
  #print(sig)
  u = matrix(rnorm(m*k), m, k)
  x = u %*% l
  #print(x[1:10, ])
  x = x * (x > 0)
  #print(x[1:10, ])
  #print(cov(x))
  df1 = data.frame(x)
  dfBu0 = BucketDf(df=df1, bucketSize=NULL, bucketNum=5, cols=NULL)
  dfBu0
  dfBu = ClustAndAssess(df=df1, centers=5, method="bucket")
  dfCl = ClustAndAssess(df=df1, centers=5, method="kmeans2")
  #print(dfBu[["withinss"]])
  #print(dfBu[["size"]])
  #dfBu0
  #dfBu
}

## plot clusters
OrderPlotClust = function(
    df0, sizes, withinPerClust, levOrd=NULL,
    yCol="y", orderIt=FALSE, pltTitle="", ord=NULL, stackPlotXlab=NULL,
    stackPlotYlab=NULL, fontSizeAlpha=1) {

  tot = rowSums(df0)

  if (orderIt) {
    if (is.null(ord)) {
      ord = order(-tot)
    }

    df0 = df0[ord, , drop=FALSE]
    sizes = sizes[ord]
    withinPerClust = withinPerClust[ord]
  }

  props = 100 * sizes / sum(sizes)
  df0["clust"] = (1:(dim(df0)[1]))

  meltDf = melt(df0, id="clust")

  if (!is.null(levOrd)) {
      lev = levOrd[length(levOrd):1]
      meltDf[ , "variable"] = factor(
          as.character(meltDf[ , "variable"]), levels = lev)
      meltDf = meltDf[order(meltDf[ , "variable"]), ]
  }

  plotSize = qplot(
      x=1:length(sizes), xend=1:length(sizes), y=0, size=I(10),
      yend=props, geom="segment", ylim=c(0, max(props)), ylab="size (%)",
      xlab="", color=I("darkgrey")) + theme(legend.position="none") +
      theme(axis.text.x = element_text(angle = 0, hjust = 0.5, face="bold",
        size=14*fontSizeAlpha)) +
      theme(axis.text=element_text(size=14*fontSizeAlpha, face="bold"),
        axis.title=element_text(size=14*fontSizeAlpha, face="bold")) +
      theme(plot.title = element_text(face="bold", size=16*fontSizeAlpha,
        hjust=0))

  plotWithinErr =  qplot(
      x=1:length(sizes), xend=1:length(sizes), y=0, size=I(10),
      yend=withinPerClust, geom="segment", ylim=c(0, 2*max(withinPerClust)),
      ylab="Clust Within Err", xlab="clust", color=I("brown")) +
      theme(legend.position="none") +
      theme(axis.text.x = element_text(angle = 0, hjust = 0.5, face="bold",
        size=14*fontSizeAlpha)) +
      theme(axis.text=element_text(size=14*fontSizeAlpha, face="bold"),
        axis.title=element_text(size=14*fontSizeAlpha, face="bold")) +
      theme(plot.title = element_text(face="bold", size=16*fontSizeAlpha,
        hjust=0))


  stackPlot = StackPlot(meltDf, xCol="clust", yCol="value", fill="variable",
    pltTitle=paste0(pltTitle), xlab=stackPlotXlab, ylab=stackPlotYlab,
    ylim=c(0, max(tot)), fontSizeAlpha=fontSizeAlpha)

  return(list(
      "plotSize"=plotSize, "plotWithinErr"=plotWithinErr,
      "stackPlot"=stackPlot, "ord"=ord))
}


OrderPlotClustCentMid = function(
    df, centers, method="kmeans", conditions=NULL, cols=NULL, levOrd=NULL,
    plotAssess=FALSE, orderIt=TRUE, yCol="y", ss=10000,
    stackPlotLabsCent=c("", ""), stackPlotLabsMid=c("", ""), fontSizeAlpha=1,
    pltTitleSuff="") {

  sliceName = ""
  if (!is.null(conditions)) {
    res = SliceDfCondition(df=df, conditions=conditions)
    df = res[["df"]]
    sliceName = res[["sliceName"]]
  }

  dfCl = ClustAndAssess(df=df, centers=centers, method=method, cols=cols,
    iter.max=20, plotIt=plotAssess, ss=ss, fontSizeAlpha=fontSizeAlpha,
    pltTitleSuff=pltTitleSuff)
  sizes = dfCl[["size"]]
  withinPerClust = dfCl[["withinPerClust"]]

  df0 = data.frame(dfCl[["centers"]])

  outCent = OrderPlotClust(
      df0, sizes=sizes, withinPerClust=withinPerClust, levOrd=levOrd,
      yCol=yCol, orderIt=orderIt,
      pltTitle=paste0(sliceName, method, " cluster weight centers", pltTitleSuff),
      stackPlotXlab=stackPlotLabsCent[1], stackPlotYlab=stackPlotLabsCent[2],
      fontSizeAlpha=fontSizeAlpha)

  df0 = data.frame(dfCl[["middles"]])

  outMid = OrderPlotClust(
      df0, sizes=sizes, withinPerClust=withinPerClust, levOrd=levOrd,
      yCol=yCol, orderIt=orderIt,
      pltTitle=paste0(sliceName, method, " cluster mid user", pltTitleSuff),
      stackPlotXlab=stackPlotLabsMid[1], stackPlotYlab=stackPlotLabsMid[2],
      ord=outCent[["ord"]], fontSizeAlpha=fontSizeAlpha)

  plotSize1 = outCent[["plotSize"]]
  plotWithinErr1 = outCent[["plotWithinErr"]]
  stackPlot1 = outCent[["stackPlot"]]

  plotSize2 = outMid[["plotSize"]]
  plotWithinErr2 = outMid[["plotWithinErr"]]
  stackPlot2 = outMid[["stackPlot"]]

  dfCl[["clustpltList"]] = list(
      "plotSize"=plotSize1, "stackPlot1"=stackPlot1, "errPlot"=plotWithinErr2,
      "stackPlot2"=stackPlot2)
  return(dfCl)
}

## plot clustering results
PlotClustResults = function(dfCl, size=c(800, 1200)) {

  #set_plot_options(width=size[1], height=size[2])

  Multiplot(
      pltList=c(dfCl[["clustpltList"]], pltList=dfCl[["corpltList"]]), ncol=2)
}

Example = function() {
  source(src)
  varNum = 6 # param
  sampleSize = 50 # param
  levOrd = c("X4", "X3", "X1", "X2", "X5", "X6") # param
  # this is to generate a variance-covariance matrix via Choleski decomp
  l = matrix(0, varNum, varNum)
  diag(l) = 1
  for (i in 2:varNum) {
    l[i, ] = c(rnorm(i-1), 1, rep(0, varNum-i))
  }
  # this (sig) is the variance-covarience matrix of the underlying dist
  # in case you like to compare
  sig = t(l) %*% l
  u = matrix(rnorm(sampleSize*varNum), sampleSize, varNum)
  x = u %*% l
  ## this is to insure the response is always non-negative
  x = x * (x >= 0)
  df1 = data.frame(x)


  clustNum = 8 # param
  dfCl = OrderPlotClustCentMid(df=df1, centers=clustNum,
    levOrd=levOrd, method="kmeans")
  dfBu = OrderPlotClustCentMid(df=df1, centers=10, method="bucket")

  PlotClustResults(dfBu)
  PlotClustResults(dfCl)
}
