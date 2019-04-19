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

## color functions with transparency
ColAlpha = function(colors, alpha=0.5) {

  r = col2rgb(colors, alpha=TRUE)
  r[4, ] = alpha * 255
  r = r / 255.0
  return(rgb(r[1, ], r[2, ], r[3, ], r[4, ]))
}


## compare the distribution of two variables
# making sure that the data are cut in the same way
DichomHist = function(
    x, mainText="", xlab="", signif=2, step=0.1,
    labelCol=ColALpha("red", 0.7), srt=75) {

  x = signif(x, signif)
  x = na.omit(x)
  x = sort(x)
  cutoffs = c(min(x), quantile(x, probs=seq(step, 1-step, step)), max(x))
  cutoffs = unique(cutoffs)
  x1 = cut(x, cutoffs, include.lowest=TRUE)
  labels = unique(x1)

  k = length(labels)
  eps = 0.1
  maxVal = max(as.numeric(100*table(x1) / sum(table(x1))))


  par(mar=c(10.1, 6.1, 5.1, 5.1))
  plot(
      1:k,
      as.numeric(100*table(x1) / sum(table(x1))),
      col=ColAlpha("blue", 0.6),
      type="h",
      lwd=20,
      xaxt="n",
      main=mainText,
      xlab="",
      ylab="Frequency (%)",
      xlim=c(1, k),
      ylim=c(0, maxVal + 2),
      cex.lab=1.5)

  #axis(2, cex.axis=1.2)
  axis(1, at=1:k, labels=FALSE, cex.axis=1.2)
  text(
      1:k, par("usr")[3]-1, labels=labels, srt=srt, pos=1,
      xpd=TRUE, cex=1.5, col=labelCol)
  title(xlab=xlab, line=9, cex.lab=1.6, family="Calibri Light")
}

TestDichomHist = function() {

  x = c(rep(0, 100), 10:100, 1000:1200)
  DichomHist(x, step=0.05, labelCol="red")
}

## makes gg box plots which zoom in and get rid of extreme outliers
ZoomedBoxP = function(df, x, y, fill) {

  p = ggplot(df, aes_string(x=x, y=y, fill=fill)) +
      geom_boxplot()

  if ("data.table" %in% class(df)) {
    values = eval(parse(text = paste0("values = df[ ,",  y, "]")))
  } else {
    values = df[ , y]
  }

  ylim = boxplot.stats(values)[["stats"]][c(1, 5)]
  # scale y limits based on ylim
  p = p + coord_cartesian(ylim=ylim*1.05)
  return(p)
}


TestZoomedBoxP = function() {

  m = 100
  x = sample(c("cat", "dog", "horse"), m, replace=TRUE)
  y = rnorm(m)
  fill = sample(c("me", "myself", "dude"), m, replace=TRUE)
  df = data.frame("x"=x, "y"=y, "fill"=fill)

  ZoomedBoxP(df=df, x="x", y="y", fill="fill")

  ZoomedBoxP(df=data.table(df), x="x", y="y", fill="fill")

}

ExampleOfGetFailure = function() {

  m = 100
  x1 = sample(c("cat", "dog", "horse"), m, replace=TRUE)
  y1 = rnorm(m)
  fill1 = sample(c("me", "myself", "dude"), m, replace=TRUE)
  df = data.frame("x"=x1, "y"=y1, "fill"=fill1)

  dt = data.table(df)

  ## get does not work!
  y = "y"
  dt[ , get(y)]

  ## get works!
  yCol = "y"
  dt[ , get(yCol)]

  ## works always; but its not pretty!
  eval(parse(text = paste0("values = dt[ ,",  y, "]")))
  eval(parse(text = paste0("values = dt[ ,",  yCol, "]")))

}

## compare the distribution of two variables
# making sure that the data are cut in the same way
PltCompareDist = function(
    x, y, mainText="", xlab="", varNames=c("var1", "var2"), signif=2,
    step=0.1) {

  x = signif(x, signif)
  y = signif(y, signif)
  z = na.omit(c(x, y))
  z = sort(z)
  cutoffs = c(min(z), quantile(z, probs=seq(step, 1-step, step)), max(z))
  cutoffs = unique(cutoffs)
  x = sort(x)
  y = sort(y)
  x1 = cut(x, cutoffs, include.lowest=TRUE)
  y1 = cut(y, cutoffs, include.lowest=TRUE)
  z1 = cut(z, cutoffs, include.lowest=TRUE)
  labels = unique(z1)

  k = length(labels)
  eps = 0.1
  maxVal = max(
      as.numeric(100*table(x1) / sum(table(x1))),
      as.numeric(100*table(y1) / sum(table(y1))))

  plot(
      1:k,
      as.numeric(100*table(x1) / sum(table(x1))),
      col=ColAlpha("blue", 0.6),
      type="h",
      lwd=10,
      xaxt="n",
      main=mainText,
      xlab=xlab,
      ylab="Frequency (%)",
      xlim=c(1, k),
      ylim=c(0, maxVal + 2),
      cex.lab=1.5)

  axis(2, cex.axis=1.2)

  lines(
      (1:k) + eps,
      as.numeric(100*table(y1) / sum(table(y1))),
      col=ColAlpha("red", 0.6),
      lwd=10,
      type="h")

  axis(1, at=1:k, labels=labels, cex.axis=1.2)
  legend(
      "topright", inset=c(0, 0), legend=varNames, lwd=c(8, 8),
      col=ColAlpha(c("blue", "red"), 0.5), title="", cex=1.3, pt.cex=1)
}

TestPltCompareDist = function() {
  x = runif(100, 1, 2)
  y = runif(100, 1.5, 2.5)
  PltCompareDist(1:10, 1:11)
}


## compare categ distbn
PltCompare_categDist = function(df, xCol, fillCol) {

  dt = data.table(df)

  freqDt = dt[ , .(num=.N), by=mget(c(xCol, fillCol))]

  freqDt2 = freqDt[ , .(total_num=sum(num)), by=mget(fillCol)]

  freqDt3 = merge(freqDt, freqDt2, by=fillCol, all.x=TRUE, all=FALSE)

  freqDt3[ , "prop"] = round(100*freqDt3[ , num] / freqDt3[ , total_num], 2)


  p = ggplot(
      data.frame(freqDt3),
      aes_string(x=xCol, y="prop", fill=fillCol)) +
      geom_bar(stat="identity", width=.5, position="dodge") + ylab("freq.") +
      xlab(xCol) +
      guides(fill=guide_legend(title=fillCol)) +
      theme(
          text=element_text(size=16),
          axis.text.x=element_text(angle=30, hjust=1))

  return(list("p"=p, "propDt"=freqDt3))
}


# plots a bivariate categorical variable
PltStack_bivarCateg = function(df, xCol, fillCol) {

  dt = data.table(df)

  freqDt = dt[ , .(num=.N), by=mget(c(xCol, fillCol))]

  freqDt[ , "prop"] = 100 * freqDt[ , num] / sum(freqDt[ , num])

  pltTitle = paste("prop wrt", xCol, "and", fillCol)

  p = StackPlot(
      meltDf=data.frame(freqDt),
      xCol=xCol,
      yCol="prop",
      fill=fillCol,
      pltTitle=pltTitle)

  return(list("p"=p, "freqDt"=freqDt))

}


## makes a histogram for d, also adds the mean and median lines for d
Plt_compareDiffWithZero = function(
    d, mainText="", xlab="", varNames=c("var1", "var2")) {

  med = median(d, na.rm=TRUE)
  m = mean(d, na.rm=TRUE)

  hist(
      d,
      col=ColAlpha("red", 0.5),
      main=paste0(mainText,": ", varNames[1], " - ", varNames[2]),
      xlab="Diff",
      probability=TRUE)

  abline(v=med, lty=2, lwd=3, col=ColAlpha("blue", 0.5))
  abline(v=m, lty=2, lwd=3, col=ColAlpha("green", 0.5))
  legend("topright", legend=c("median", "mean"), lty=c(1, 1), lwd=c(5, 5),
    col=ColAlpha(c("blue","green"), 0.5), cex=0.8)
}

TestPlt_compareDiffWithZero = function() {
  d = rnorm(100)
  Plt_compareDiffWithZero(d)
}

## comparing a valueCol boxplots across categories
Plt_compareBoxPlot = function(df, compareCol, valueCol, pltTitle="") {
  p = (
      ggplot(df, aes_string(compareCol, valueCol, fill=compareCol)) +
      geom_boxplot() + labs(title=compareCol) +  xlab(compareCol) +
      ylab(valueCol) + ggtitle(pltTitle))

  return(p)
}

Plt_compareDensity = function(
    df, compareCol, valueCol, addMeans=TRUE, pltTitle="") {

  df = df[ , c(compareCol, valueCol)]
  meanDf = data.frame(
      data.table(df)[ , lapply(.SD, mean, na.rm=TRUE), by=compareCol])
  colnames(meanDf) = c(compareCol, valueCol)
  print(meanDf)
  p = (
      ggplot(df, aes_string(x=valueCol, fill=compareCol, color=compareCol))
      + geom_density(alpha=0.3, size=2)
      + labs(title=compareCol)
      + xlab(compareCol)
      + ylab(valueCol)
      + ggtitle(pltTitle))

  if (addMeans) {
    p = p + geom_vline(
        data=meanDf,
        aes_string(xintercept=valueCol, color=compareCol),
        linetype="dashed",
        size=1, alpha=0.5)
  }

  return(p)
}

## plotting multiple plots in ggplots
Multiplot = function(pltList=NULL, ncol=NULL) {

  # Make a list from the ... arguments and pltList
  numPlots = length(pltList)

  # Make the panel
  # Number of columns of plots
  if (is.null(ncol)) {
    ncol = ceiling(sqrt(numPlots))
  }

  do.call(what=function(...) {grid.arrange(..., ncol=ncol)}, pltList)

  #pltCols = cols
  #pltRows = ceiling(numPlots / pltCols)
  # Set up the page
  #grid.newpage()
  #pushViewport(viewport(layout=grid.layout(pltRows, pltCols)))
  #vplayout = function(x, y) {
  #  viewport(layout.pos.row=x, layout.pos.col=y)
  #}

  # Make each plot, in the correct location
  #for (i in 1:numPlots) {
  #  curRow = ceiling(i / pltCols)
  #  curCol = (i-1) %% pltCols + 1
  #  print(plots[[i]], vp=vplayout(curRow, curCol), newpage=FALSE)
  #}
}

## saving multiple plots using ggsave in one page
GgsaveMulti = function(
    fn,
    pltList,
    ncol=NULL,
    Device=function(...)Cairo::CairoPNG(..., units="in", dpi=120),
    width=6,
    height=6) {

  if (is.null(ncol)) {
    ncol = round(sqrt(length(pltList)))
  }
  grd = do.call(gridExtra::arrangeGrob, c(pltList, ncol=ncol))

  ggplot2::ggsave(
      fn,
      grd,
      width=width,
      height=height,
      device=Device)
  dev.off()
  close(fn)
}

TestGgsaveMulti = function() {

  plt1 = ggplot2::ggplot(iris, aes(Species)) +
    ggplot2::geom_bar()

  plt2 =  ggplot2::ggplot(iris, aes(Sepal.Width, Sepal.Length)) +
    ggplot2::geom_point()

  pltList = list(plt1, plt2)
  fn = "test.png"
  GgsaveMulti(
      fn=file(fn, "w"),
      pltList=pltList
  )
}

## it plots multiple lines in the same plot
# xCols are the x-axis columns
# yCols are the y-axis columns
# if x-axis variable are the same for all lines, xCols can be passed as one var
# if y-axis variable are the same for all lines, yCols can be passed as one var
PltDfColsLines = function(
    df, xCols, yCols, ylim=NULL, xlim=NULL,
    xlab=NULL, ylab=NULL, main="",
    legend=NULL, legPos="topleft", lwd=2.5, cex=1.5,
    cex.main=1.5, cex.axis=1.6, cex.lab=1.5, cex.legend=1.5,
    varLty=FALSE, sizeAlpha=1) {

  n_x = length(xCols)
  n_y = length(yCols)
  k = max(n_x, n_y)

  if (n_x != n_y && n_x != 1 && n_y != 1) {
    warnings(
        "length of the xCols, yCols is not the same. Also none is 1.")
    return(NULL)
  }

  if (n_x == 1) {
    xCols = rep(xCols, k)
  }
  if (n_y == 1) {
    yCols = rep(yCols, k)
  }

  if (is.null(ylab)) {
    ylab = paste(unique(yCols), collapse=";")
  }

  if (is.null(xlab)) {
    xlab = paste(unique(xCols), collapse=";")
  }

  if (is.null(legend)) {
    legend = yCols
  }

  if (is.null(ylim)) {
    yMax = max(df[ , yCols], na.rm=TRUE)
    yMin = min(df[ , yCols], na.rm=TRUE)
    yMax = yMax + (yMax - yMin) / 2
    ylim = c(yMin, yMax)
  }

  if (is.null(xlim)) {
    xMax = max(df[ , xCols], na.rm=TRUE)
    xMin = min(df[ , xCols], na.rm=TRUE)
    xlim = c(xMin, xMax)
  }

  colors = rainbow(k)

  ltyVec = rep(1, k)
  if (varLty) {
    ltyVec = 1:k
  }

  plot(
      df[ , xCols[1]], df[ , yCols[1]],
      ylim=ylim, xlim=xlim,
      type="l", col=ColAlpha(colors[1], 0.75),
      ylab=ylab, xlab=xlab,
      lwd=lwd*sizeAlpha, main=main, cex.main=cex.main*sizeAlpha,
      cex=cex*sizeAlpha, cex.axis=cex.axis*sizeAlpha,
      cex.lab=cex.lab*sizeAlpha, lty=1)

  if (k > 1) {
    for (i in 2:k) {
      lines(
          df[ , xCols[i]], df[ , yCols[i]], col=ColAlpha(colors[i], 0.75),
          lwd=lwd, lty=ltyVec[i])
    }
  }

  legend(
      legPos, legend=legend, col=colors, cex=cex.legend*sizeAlpha,
      lty=ltyVec, lwd=lwd*sizeAlpha)
}

## plot bands given a lower column and an upper column: usiful for CIs
PltBands = function(
    x, yLower, yUpper, col=ColAlpha("grey", 0.5),
    border=NA, angle=NULL, density=NULL, lwd=3) {

  polygon(
      c(rev(x), x), c(rev(yLower), yUpper), col=col,
      border=NA, angle=angle, density=density, lwd=lwd)
}

TestPltBands = function() {

  plot(-20:20, -20:20)
  x = sort(rnorm(100))
  yLower = x - 6 + rnorm(100)
  yUpper = x + 6 + rnorm(100)

  polygon(
    c(rev(x), x), c(rev(yLower), yUpper),
    col=ColAlpha("blue", 0.5), border=NA,
    lwd=2, angle=45, density=20)

  legend(
      "top", legend=1, ncol=1, fill=TRUE, col=1, angle=45, density=20)
}

## it compares the CI from different methods (groups)
# across xCol (e.g. sample size)
# the data is given in long format i.e the groups data are stacked
Plt_compareCiGroups = function(
    df, xCol, lowerCol, upperCol, compareCol, compareValues=NULL,
    xlab=NULL, ylab="", main="", lwd=3, addMidPoint=TRUE) {

  yMin = min(df[ , c(lowerCol, upperCol)], na.rm=TRUE)
  yMax = max(df[ , c(lowerCol, upperCol)], na.rm=TRUE)
  yMax = yMax + (yMax - yMin) / 3

  xMin = min(df[ , xCol], na.rm=TRUE)
  xMax = max(df[ , xCol], na.rm=TRUE)

  if (is.null(xlab)) {
    xlab = xCol
  }

  plot(
      x=c(xMin, xMax), y=c(yMin, yMax), type="n", xlab=xlab,
      ylab=ylab, main=main, cex.main=1.5, cex.axis=1.2, cex.lab=1.2)

  if (is.null(compareValues)) {
    compareValues = unique(df[ , compareCol])
  }

  k = length(compareValues)
  angles = (1:k)*180 / (k + 1)
  cols = rainbow(k)
  for (i in 1:k) {
    value = compareValues[i]
    df0 = df[df[ , compareCol] == value, ]
    PltBands(
        x=df0[ , xCol], yLower=df0[ , lowerCol], yUpper=df0[ , upperCol],
        col=ColAlpha(cols[i], 0.3), angle=angles[i], density=60, lwd=lwd)
  }

  legend(
      "top", legend=compareValues, col=cols, lwd=lwd,  bty="n",
       angle=angles, density=60, cex=1.2)

  dt = data.table(df)
  aggDt = DtSimpleAgg(
      dt=dt, valueCols=c(lowerCol, upperCol), gbCols=compareCol, F=mean)
  aggDt[ , "midPoint"] = (aggDt[ , get(lowerCol)] + aggDt[ , get(upperCol)]) / 2

  if (addMidPoint) {

    for (i in 1:k) {
      value = compareValues[i]
      df0 = df[df[ , compareCol] == value, ]
      midPoint = aggDt[(aggDt[ , get(compareCol)] == value), midPoint]
      x = df0[ , xCol]
      points(
          x=x, y=rep(midPoint, length(x)), col=ColAlpha(cols[i], 0.5),
          pch=10, cex=0.3)
    }
  }
}

TestPlt_compareCiGroups = function() {

  x = seq(0, 1, 0.01)
  yLower = sin(2 * pi * x) - 0.1
  yUpper = sin(2 * pi * x) + 0.1
  group = "1"
  df1 = data.frame(x, yLower, yUpper, group)

  yLower = cos(2 * pi * x) - 0.1
  yUpper = cos(2 * pi * x) + 0.1
  group = "2"
  df2 = data.frame(x, yLower, yUpper, group)
  df = rbind(df1, df2)

  Plt_compareCiGroups(
      df=df, xCol="x", lowerCol="yLower", upperCol="yUpper",
      compareCol="group", lwd=3)
}

## plots selected columns of df (cols) as side by side bars in a plot,
# it uses different color per col
Plt_barsSideBySide = function(
    df, cols=NULL, legendPosition="topright",
    xLabels=NULL, xlim=NULL, ylim=NULL, xlab=NULL, ylab=NULL,
    legendLabels=NULL, colors=NULL, xpd=FALSE) {

  if (is.null(cols)) {
    cols = colnames(df)
  }

  df = df[ , cols, drop=FALSE]
  n = dim(df)[2]

  if (is.null(colors)) {
    colors = rainbow(n)
  }

  if (is.null(legendLabels)) {
    legendLabels = cols
  }

  if (is.null(ylim)) {
    ylim = c(min(df, na.rm=TRUE), max(df, na.rm=TRUE))
  }

  if (is.null(xlab)) {
    xlab = ""
  }

  if (is.null(ylab)) {
    ylab = ""
  }


  par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=xpd)
  eps = 3/(4*n)
  lwd = 100/n
  x = df[ , 1]
  if (is.null(xlim)) {
    xlim = c(1, length(x) + 1)
  }

  par(font.axis = 2)
  plot(
      1:length(x), x, col=ColAlpha(colors[1], 0.5),
      xlim=xlim, ylim=ylim, type="h", lwd=lwd, xlab=xlab,
      ylab=ylab, xaxt = "n")

  if (is.null(xLabels)) {
    xLabels = 1:length(x)
  }

  for (i in 2:n) {
    y = df[ , i]
    lines(
        1:length(y) + (i-1)*eps, y, col=ColAlpha(colors[i], 0.5) ,
        xlim=xlim, ylim=ylim, type="h", lwd=lwd, xlab=xlab, ylab=ylab)
  }

  legend(
      legendPosition, inset=rep(0, n), legend=legendLabels,
      lwd=rep(80/(n), n), col=ColAlpha(colors, 0.5), title="")
  axis(1, at=1:length(x), labels=xLabels, las=2, cex.axis=0.95)
}

TestPlt_barsSideBySide = function() {
  df0 = data.frame("x"=1:10, "y"=2:11)
  Plt_barsSideBySide(df0, legendPosition="topleft")
}

## plots resp wrt wrtCol
## while it slices wrt groupCol
# if there are multilple values for each c(wrtCol, groupCol), it aggregates
PlotWrtGroup = function(
    df, resp, wrtCol, groupCol,
    AggF=function(x){mean(x, na.rm=TRUE)}, group=NULL,
    main="", ColPattern=rainbow, lwd=4, lty=NULL, xlab=NULL, ylab=NULL,
    alpha=0.5, gridAlpha=0.2, gridLwd=2) {


  df2 = df[ , c(resp, wrtCol, groupCol)]
  formulaText = paste0(resp, '~', wrtCol, '+', groupCol)
  formula = as.formula(formulaText)
  dfAgg = aggregate(formula, data=df2, FUN=AggF)

  if (is.null(group)) {
    group = unique(df[ , groupCol])
  }

  yMax = max(dfAgg[ , resp], na.rm=TRUE)
  yMin = min(dfAgg[ , resp], na.rm=TRUE)
  delta = (yMax - yMin)/5
  yMax = yMax + delta
  yMin = yMin - delta

  colors = rainbow(length(group))
  if (is.null(lty)) {
    lty = rep(1, length(group))
  }

  if (is.null(xlab)) {
    xlab = wrtCol
  }

  if (is.null(ylab)) {
    ylab = resp
  }

  #c(bottom, left, top, right)
  par(mar=c(5.1, 4.1, 4.1, 14.1), xpd=TRUE)
  par(font=2)

  for (i in 1:length(group)) {
    g = group[i]
    dfAgg2 = dfAgg[dfAgg[ , groupCol] == g, ]
    if (i == 1) {
      plot(
          dfAgg2[ , wrtCol], dfAgg2[ , resp],
          col=ColAlpha(colors[i], alpha), main=main,
          ylim=c(yMin, yMax), ylab=resp, xlab=wrtCol, type='l', cex.main=2,
          lwd=lwd, lty=lty[i], cex.lab=1.5, cex.axis=1.5, font.lab=2)

    } else {
      lines(
          dfAgg2[ , wrtCol],
          dfAgg2[ , resp],
          col=ColAlpha(colors[i], alpha),
          lwd=lwd, lty=lty[i])
    }
  }

  legend("topright", inset=c(-0.2, 0), legend=group,
    lwd=rep(lwd, length(group)), title=groupCol,
    col=ColAlpha(colors, alpha), cex=1, lty=lty)
  par(xpd=FALSE)
  grid(nx=NULL, ny=NULL, col=ColAlpha('black', gridAlpha), lwd=gridLwd)
}

## create stackplot with ggplot
StackPlot = function(
    meltDf, xCol, yCol, fill, xlab=NULL, ylab=NULL, labelCol=NULL,
    xAxisRotation=0, manualPalette=TRUE, colors=NULL, palette="Paired",
    legendTitle="variable", pltTitle=NULL,
    ylim=NULL, fontSizeAlpha=1) {

  if (is.null(pltTitle)) {
    pltTitle = paste(yCol, "wrt", xCol, "partitioned by", fill)
  }

  if (is.null(xlab)) {
    xlab = xCol
  }

  if (is.null(ylab)) {
    ylab=ylab
  }

  p = ggplot(meltDf, aes_string(x=xCol, y=yCol, fill=fill)) +
    xlab(xlab) + ylab(ylab) +
    geom_bar(stat="identity", colour=ColAlpha("black", 0.3), lwd=0.5) +
    ggtitle(pltTitle) +
    scale_y_continuous(limits=ylim) +
    guides(
        fill=guide_legend(override.aes=list(colour=ColAlpha("black", 0.5))),
        lwd=1) +
    guides(colour=FALSE) +
    theme(
        axis.text.x=element_text(
            angle=xAxisRotation,
            hjust=0.5, face="bold", size=14*fontSizeAlpha)) +
    theme(
        axis.text=element_text(
            size=14*fontSizeAlpha, face="bold"),
        axis.title=element_text(size=14*fontSizeAlpha, face="bold")) +
    theme(plot.title=element_text(
        face="bold", size=16*fontSizeAlpha, hjust=0)) +
    guides(
        fill=guide_legend(
          keywidth=(2.5)*fontSizeAlpha, keyheight=(2.5)*fontSizeAlpha)) +
    theme(legend.text=element_text(size=12*fontSizeAlpha)) +
    guides(fill=guide_legend(title=legendTitle)) +
    guides(colour=guide_legend(title.hjust=2*fontSizeAlpha)) +
    guides(colour=guide_legend(override.aes=list(size=20*fontSizeAlpha)))

    if (manualPalette) {
      if (is.null(colors)) {
        n = length(unique(meltDf[ , fill]))
        colors = GenColors2(n)
      }
      p = p + scale_color_manual(values=colors)
    } else {
      p = p + scale_fill_brewer(palette=palette)
    }

    if (!is.null(labelCol)) {
      p = p + geom_text(aes_string(
          label=labelCol),
          position=position_stack(vjust=0.5),
          size=5*fontSizeAlpha)
    }

  p = p + theme(
      text=element_text(size=16),
      axis.text.x=element_text(angle=30, hjust=1))
  return(p)
}

## creates level plots
LevPlot = function(
    df, mainText="", at=seq(-1, 1, 0.01), colRange=c("yellow", "blue")) {
  #library(ggplot2)
  rgb.palette = colorRampPalette(colRange, space = "rgb")
  p = levelplot(
      df, main=mainText, xlab="", ylab="",
      col.regions=rgb.palette(length(at)), cuts=length(at), at=at,
      scales=list(x=list(rot=90)))
  return(p)
}

## level plot with ggplot library
LevPlotGg = function(
    df, mainText="", limits=NULL, colRange=c("yellow", "grey", "blue"),
    xlab=NULL, ylab=NULL, xLabelsAngel=90, fontSizeAlpha=1) {

  meltedDf = melt(df)
  #myPalette = colorRampPalette(rev(brewer.pal(11, "Spectral")))
  #sc = scale_fill_gradientn(colors = myPalette(100), limits=c(-1, 1))
  #sc = scale_fill_gradientn(colors = myPalette(100), low="yellow", high="blue", limits=c(-1, 1))
  #sc = scale_fill_distiller(palette = "Spectral", low=-1, high=1)
  #sc = scale_colour_gradient(limits=c(-1, 1), low="red", high="white")
  #sc = scale_fill_distiller(palette = palette)
  if (is.null(limits)) {
    limits = c(
        min(meltedDf[ , "value"], na.rm=TRUE),
        max(meltedDf[ , "value"], na.rm=TRUE))
  }

  sc = scale_fill_gradientn(limits=limits, colours=colRange)
  p = (
      ggplot(data=meltedDf, aes(x=Var1, y=Var2, fill=value)) +
      geom_tile(stat="identity",  colour=ColAlpha("black", 0.2), lwd=0.4) +
      ggtitle(mainText) +
      theme(axis.text.x=element_text(angle=xLabelsAngel, hjust=1, vjust=1)) +
      sc +
      theme(
          axis.text=element_text(size=18*fontSizeAlpha, face="bold"),
          axis.title=element_text(size=18*fontSizeAlpha, face="bold")) +
      theme(
          plot.title=element_text(
              face="bold", size=20*fontSizeAlpha, hjust=0)) +
      guides(colour=FALSE) +
      theme(legend.text=element_text(size=15*fontSizeAlpha)))
  #theme(panel.background = element_rect(colour = "black")) +
  #guides(fill = guide_legend(override.aes = list(colour =  ColAlpha("black", 0.5)))) +

  if (!is.null(xlab)) {
    p = p + labs(x = xlab)
  }

  if (!is.null(ylab)) {
    p = p + labs(y = ylab)
  }

  return(p)
}

## make 2d correlation plot
CorPlt = function(
    df, cols=NULL, subsetIndList=NULL, colRange=c("red", "white", "green"),
    tightLimits=FALSE, limits=c(-1, 1), xLabelsAngel=90,
    mainText="", fontSizeAlpha=1) {

  if (!is.null(cols)) {
    df = df[ , cols, drop=FALSE]
  }

  dfCor = cor(df, use="complete")
  if (!is.null(subsetIndList)) {
    dfCor = dfCor[subsetIndList[[1]], subsetIndList[[2]]]
  }

  if (tightLimits) {
    limits = c(min(dfCor), max(dfCor))
  }

  p = LevPlotGg(
      df=dfCor, mainText=mainText, limits=limits, colRange=colRange,
      xlab="", ylab="", xLabelsAngel=xLabelsAngel, fontSizeAlpha=fontSizeAlpha)
  #p = LevPlot(dfCor, mainText=mainText, colRange=colRange)
  return(p)
}

## creates corr plots
CorPerSlice = function(
    df, cols=NULL, conditions=NULL, colRange=c("yellow", "grey", "blue"),
    mainText="", fontSizeAlpha=1) {

  sliceName = ""
  if (!is.null(conditions)) {
    res = SliceDfCondition(df=df, conditions=conditions)
    df = res[["df"]]
    sliceName = res[["sliceName"]]
  }

  out = CorPlt(
      df, cols=cols, colRange=colRange, mainText=paste(mainText, sliceName),
      fontSizeAlpha=fontSizeAlpha)
  return(out)
}

## create a continuous variable set correlation plot
# and latex table and save them
SaveCorPlt_andLatex = function(
    df, subsetIndList=NULL, tightLimits=FALSE, limits=c(-1, 1),
    colRange=c("red", "white", "green"), xLabelsAngel=90,
    figsPath, tablesPath, fnSuffix, cropColsNum=NULL, tableSize=NULL) {

  fn0 = paste0(figsPath, "cor_", fnSuffix, ".png")
  Mark(fn0, "filename")
  fn = file(fn0, "w")
  r = 1.9
  Cairo(
        width=850*r, height=480*r, file=fn, type="png", dpi=120*r,
        pointsize=12*r)

  print(CorPlt(
      df=df, subsetIndList=subsetIndList, tightLimits=tightLimits,
      limits=limits, colRange=colRange, xLabelsAngel=xLabelsAngel))
  dev.off()
  close(fn)

  cropColsNum = 3
  corMat = round(cor(df), 2)
  corMatCropped = corMat
  if (!is.null(cropColsNum)) {
    corMatCropped = corMat[ , 1:min(cropColsNum, dim(corMat)[1])]
  }

  corLat = xtable(
      corMatCropped,
      caption=gsub("_", " ", x=paste(fnSuffix, "corr")),
      label=paste0(fnSuffix, "corr", sep="-"))

  if (!is.null(tableSize)) {
    corLat = gsub("\\centering", paste("\\centering ", tableSize), x=corLat)
  }
  fn0 = paste0("cor_", fnSuffix, ".tex")
  fn0 = paste0(tablesPath, fn0)
  fn0 = tolower(fn0)
  Mark(fn0, "filename")
  fn = file(fn0, "w")
  print(x=corLat, file=fn)
  close(fn)

  return(list(
      "corMat"=corMat, "corLat"=corLat, "corMatCropped"=corMatCropped))
}

## save both plot and latex table in the specified paths
SaveCorPlt_andLatex_andShow = function(
    df, subsetIndList, label, fnSuffix, figsPath,
    tablesPath, tightLimits=FALSE, limits=c(-0.05, 0.7),
    colRange=c("white", "yellow", "green")) {

  fnSuffix2 = paste0(fnSuffix, label)
  res = SaveCorPlt_andLatex(
      df=df, subsetIndList=subsetIndList,  tightLimits=tightLimits,
      limits=limits, colRange=colRange, xLabelsAngel=45,
      figsPath=figsPath, tablesPath=tablesPath, fnSuffix=fnSuffix2,
      cropColsNum=3, tableSize=NULL)

  out = CorPlt(
      df=df, subsetIndList=subsetIndList, tightLimits=tightLimits,
      limits=limits, colRange=colRange, xLabelsAngel=45)

  return(out)
}

## generate colors
GenColors = function(n) {

  library(RColorBrewer)
  ind = brewer.pal.info[ , "category"] == "qual"
  qualColPals = brewer.pal.info[ind, ]
  colVec = unlist(mapply(
      brewer.pal,
      qualColPals[ , "maxcolors"],
      rownames(qualColPals)))

  return(colVec)
}

TestGenColors = function() {
 n = 64
 colVec = GenColors(n)
 pie(rep(1, n), col=sample(colVec, n))
}

GenColors2 = function(n, pals=NULL) {

  df = data.frame(
    "pal"=c(
        "Accent", "Dark2", "Paired", "Pastel1",
        "Pastel2", "Set1", "Set2", "Set3"),
    "num"=c(8, 8, 12, 9, 8, 9, 8, 12))

  if (!is.null(pals)) {
    df = df[df[ , "pal"] %in% pals, ]
  }

  colors = NULL

  for (i in 1:nrow(df)) {
    colors = c(
        colors,
        brewer.pal(n=df[i, "num"], name=as.character(df[i, "pal"])))
  }

  m = length(colors)
  if (m < n) {
    warning(paste("n is larger than the number of avail cols: ", m))
      return(NULL)
  }

  return(sample(x=colors, size=n, replace=FALSE))
}

TestGenColors2 = function() {

  n = 20
  colVec = GenColors2(n)

  pie(rep(1, n), col=sample(colVec, n))
}

## Plots a value column (valueCol) vs xCol wrt two grouping:
# first grouping (spec. by splitCurveCol) is done using
# a different color for each element
# second grouping (spec. by splitPanelCol) is done by using a different panel
Plt_splitCurve_splitPanel = function(
    df, xCol, valueCol, splitCurveCol, splitPanelCol=NULL,
    errBars=NULL, xlab=NULL, ylab=NULL, type="line", size=2,
    remove_xAxis=FALSE, pltTitle=NULL, savePlt=FALSE,
    fileName=NULL, fnLabel=NULL, figsPath="") {

  if (is.null(pltTitle)) {
    pltTitle = gsub("_", " ", valueCol)
  }

  if (is.null(xlab)) {
    xlab = gsub("_", " ", xCol)
  }

  if (is.null(ylab)) {
    ylab = gsub("_", " ", valueCol)
  }

  geom_custom = geom_line
  if (type == "points") {
    geom_custom = geom_point
  }

  p = (
      ggplot(df, aes(get(xCol), get(valueCol))) +
      geom_custom(aes(color=get(splitCurveCol)), size=size) +
      scale_y_continuous() +
      theme_bw() +
      xlab(xlab) +
      ylab(ylab) +
      theme(
          axis.text.x=element_text(size=14),
          axis.text.y=element_text(size=14),
          legend.position="top",
          legend.title=element_blank(),
          legend.text=element_text(size=14),
          strip.text.x=element_text(size=14, face="bold"),
          plot.title=element_text(size=18, face="bold", hjust=0.5)) +
      ggtitle(pltTitle)
  )

  if (remove_xAxis) {
    p = p + theme(
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
  }

  if (!is.null(splitPanelCol)) {
    p = p + facet_wrap(~get(splitPanelCol))
  }

  if (!is.null(errBars)) {
    p = p + geom_errorbar(errBars, width=.3)
  }

  if (savePlt) {

    if (is.null(fileName)) {
      fileName = paste0(
          valueCol, "_foreach_", xCol, "_foreach_", splitCurveCol)
      ## is an extra label is to be added to the figure, we do so here
      if (!is.null(fnLabel)) {
        fileName = paste0(fileName, "_", fnLabel)
      }

      fileName = paste0(fileName, ".png")
    }

    print(paste("file name:", fileName))
    fn = paste0(figsPath, fileName)
    print(paste("file name with path:", fn))
    #print(fn)
    p + ggsave(file=fn, width=10, height=6)
  }

  return(p)
}

## Test function for above
TestPlt_splitCurve_splitPanel = function() {

  n = 100

  gender = sample(c("men", "women"), n, replace=TRUE)
  country = sample(c("uk", "us", "jp"), n, replace=TRUE)

  genderEffect = list("men"=1, "women"=-5)
  countryEffect = list("uk"=0, "us"=2, "jp"=-2)

  F = function(x) {
    genderEffect[[x[1]]] + countryEffect[[x[2]]]
  }

  df = data.frame("gender"=gender, "country"=country)

  df[ , "x"] = 1:n
  df[ , "y"] = unlist(apply(X=df, MARGIN=1, FUN=F)) + 2 * df[ , "x"]

  Plt_splitCurve_splitPanel(
      df=df, xCol="x", valueCol="y", splitCurveCol="gender",
      splitPanelCol="country", type="line", savePlt=FALSE)
}

## heatmap of value column vs xCol values
# and across two possible groupings
# grouping 1: specified in splitVertCol which is mapped across y-axis
# grouping 2 (optional): specified in splitPanelCol and used to split plots
# into various panels
HeatMap_splitVertCol_splitPanel = function(
    df, xCol, splitVertCol, valueCol, splitPanelCol=NULL,
    breaks=NULL, xlab=NULL, ylab=NULL, pltTitle=NULL,
    scaleFillValues=NULL,
    savePlt=FALSE, fileName=NULL, fnLabel=NULL, figsPath="") {

  if (is.null(pltTitle)) {
    pltTitle = gsub("_", " ", valueCol)
  }

  if (is.null(xlab)) {
    xlab = gsub("_", " ", xCol)
  }

  if (is.null(ylab)) {
    ylab = gsub("_", " ", splitVertCol)
  }

  if (is.null(scaleFillValues)) {
    scaleFillValues = 1:n

  }

  if (class(df[ , valueCol]) %in% c("character", "factor")) {
    n = length(unique(df[ , valueCol]))
    scaleFill = scale_fill_manual(values=ColAlpha(scaleFillValues, 0.95))
  } else {
    scaleFill = scale_fill_gradientn(
        colours=c(ColAlpha("grey", 0.75), "yellow", "red"), na.value="black")
  }

  p = (
      ggplot(df, aes_string(xCol, splitVertCol)) +
      geom_tile(
          aes_string(fill=valueCol),
          colour=ColAlpha("grey", 0.1),
          lwd=0.4) +
      xlab(xlab) +
      ylab(ylab) +
      scaleFill +
      guides(fill=guide_legend(title="")) +
      theme_bw() +
      theme(
          axis.text.x=element_text(size=14),
          axis.text.y=element_text(size=10),
          strip.text.x=element_text(size=14, face="bold"),
          plot.title = element_text(size=18, face="bold", hjust=0.5),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
      ggtitle(pltTitle))

    if (!is.null(breaks)) {
      p = p + scale_x_discrete(breaks=breaks)
    }

    if (!is.null(splitPanelCol)) {
      p = p + facet_wrap(~get(splitPanelCol))
    }

  if (savePlt) {

    if (is.null(fileName)) {
      fileName = paste0(valueCol, "_foreach_", xCol, "_across_",  yCol)

      ## is an extra label is to be added to the figure, we do so here
      if (!is.null(fnLabel)) {
        fileName = paste0(fileName, "_", fnLabel)
      }

      fileName = paste0(fileName, ".png")
    }

    print(paste("file name:", fileName))
    fn = paste0(figsPath, fileName)
    p + ggsave(file=fn, width=10, height=6)

  }

  return(p)
}

TestHeatMap_splitVertCol_splitPanel = function() {

  n = 10000
  app = sample(c("fb", "ch", "agsa", "wa", "sc"), n, replace=TRUE)
  country = sample(c("uk", "us", "jp"), n, replace=TRUE)

  genderEffect = list("fb"=1, "ch"=-15, "agsa"=10, "wa"=2, "sc"=9)
  countryEffect = list("uk"=-20, "us"=20, "jp"=-9)

  F = function(x) {
    genderEffect[[x[1]]] + countryEffect[[x[2]]]
  }

  df = data.frame("app"=app, "country"=country)
  df[ , "x"] = sample(1:10, size=n, replace=TRUE)
  df[ , "value"] = unlist(apply(X=df, MARGIN=1, FUN=F)) + 2 * df[ , "x"]

  HeatMap_splitVertCol_splitPanel(
    df=df, xCol="x", splitVertCol="app", valueCol="value",
    splitPanelCol="country")


  ## categorical response
  df = data.frame(
      person=factor(paste0("id ", 1:50),
      levels =rev(paste0("id ", 1:50))),
      matrix(sample(LETTERS[1:3], 150, TRUE), ncol=3))

  df2 = melt(df, id.var="person")

  HeatMap_splitVertCol_splitPanel(
      df=df2, xCol="variable", splitVertCol="person", valueCol="value",
      splitPanelCol=NULL)
}

## plot a bar chart for value column for each categ given in a categ column
# add actual values close the bars
BarChart_valueAdded = function(
    df, categCol, valueCol, xlab=NULL, ylab=NULL, rounding=2,
    pltTitle=NULL, savePlt=FALSE, fileName=NULL, fnLabel=NULL, figsPath="") {

  if (is.null(pltTitle)) {
    pltTitle = gsub("_", " ", valueCol)
  }

  if (is.null(xlab)) {
    xlab = gsub("_", " ", categCol)
  }

  if (is.null(ylab)) {
    ylab = gsub("_", " ", valueCol)
  }

  dodge = position_dodge(width=0.9)
  p = (
      ggplot(
          df,
          aes(
              reorder(get(categCol), get(valueCol)),
              get(valueCol),
              fill=get(valueCol))) +
      geom_bar(stat="identity", position=dodge) +
      scale_color_discrete(name="") +
      scale_y_continuous() +
      theme_bw() +
      xlab(xlab) +
      ylab(ylab) +
      geom_text(
          aes(y=get(valueCol), label=round(get(valueCol), rounding)),
          position='stack',
          hjust=-0.5,
          vjust=0,
          color="black",
          size=4) +
      theme(
          axis.text.x=element_text(size=14),
          axis.text.y=element_text(size=14),
          strip.text.x=element_text(size=14, face="bold"),
          plot.title = element_text(size=18, face="bold", hjust=0.5),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          legend.position="none") +
      ggtitle(pltTitle) +
      coord_flip()
  )

  if (savePlt) {

    if (is.null(fileName)) {
      fileName = paste0(valueCol, "_foreach_", categCol)

      ## is an extra label is to be added to the figure, we do so here
      if (!is.null(fnLabel)) {
        fileName = paste0(fileName, "_", fnLabel)
      }

      fileName = paste0(fileName, ".png")

    }

    print(paste("file name:", fileName))
    fn = paste0(figsPath, fileName)
    p + ggsave(file=fn, width=10, height=6)

  }

  return(p)
}

## Quick plot save
QuickPltSave = function(p, fn, r=1.5) {

  fn = file(fn, "w")

  Cairo(
      width=640*r, height=480*r, file=fn, type="png", dpi=120*r,
      pointsize=10*r)

  print(p)

  dev.off()
  close(fn)
}
