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

## functions to assess rater consistency

##### PART I
#library('psych')
#library('irr')

## creating data for the irr package
Create_consDataIrr = function(data, unitCol, raterCol, resp) {

	library('reshape')
	data2 = data[ , c(unitCol, raterCol, resp)]
	formulaString = paste(unitCol, '~', raterCol)
	data3 = cast(data2, as.formula(formulaString), mean, value = resp)
	data4 = data3[ , -1]

	data4[is.na(data4)] = NA
	return(data4)
}

## simulating ratings for given units
SimulRatings = function(
		unitNum=200, raterNum=30,
		raterMeanSampler=function(n)runif(n, -1, 1),
		unitMeanSampler=function(n)runif(n, 0, 1),
		errDist=function(x)rnorm(1, mean=x),
		raterNumRange=c(2, 3, 4),
		unitCol='unit', raterCol='rater', resp='resp') {

  ## simulates ratings:  rating = raterMean + unitMean + error
  # raterMeanSampler: simulates the mean ratings for a rater
  # unitMeanSampler: simulates unit centers
  unitIds = paste0('u', 1:unitNum)
	raterIds = paste0('r', 1:raterNum)
	unitMeans = unitMeanSampler(unitNum)
	raterMeans = raterMeanSampler(raterNum)
	raterNumPerUnit = sample(raterNumRange, unitNum, replace=TRUE)
	## u is a unit number which is from 1 to unitNum
	F = function(unit) {

		## sample(n, 1) has undesired behavior that it does sample(1:n, 1)
		unitRaterNum = sample(c(raterNumRange, raterNumRange), 1)
		out = data.frame(
			  unitCol=rep(NA, unitRaterNum),
			  raterCol=rep(NA, unitRaterNum),
			  resp=rep(NA, unitRaterNum))

		unitRaters = sample(1:raterNum, unitRaterNum, replace=FALSE)

		for (i in 1:unitRaterNum) {
			rater = unitRaters[i]
			param = raterMeans[rater] + unitMeans[unit]
			y = errDist(param)
			out[i, ] = c(unitIds[unit], raterIds[rater], y)
		}

		names(out) = c(unitCol, raterCol, resp)
		return(out)
	}

	dfList = lapply(X=1:unitNum, FUN=F)
	df = do.call("rbind", dfList)
	df[ , resp] = as.numeric(df[ , resp])
	return(df)
}

Example = function() {

	res = SimulRatings()

	Fcn = function(param='') {
		data = SimulRatings(unitNum=20, raterNum=30,
			raterMeanSampler=function(n)runif(n, 0, 0),
			unitMeanSampler=function(n)runif(n, 0, 0),
			errDist=function(x)rnorm(1, mean=x),
			raterNumRange=20,
			unitCol='unit', raterCol='rater', resp='resp')

		aggData = aggregate(resp~unit, data, FUN=mean)

		S = function(x) {
			sum((x - mean(x))^2)
		}

		sData = aggregate(resp~unit, data, FUN=S)
		s = S(data[ , 'resp'])
		sAgg = S(aggData[ , 'resp'])

		b = (s == sum(sData[ , 'resp']) + dim(aggData)[1]*sAgg)
		err1 = sqrt(s)/dim(data)[1]
		err2 = sqrt(sAgg)/dim(aggData)[1]
		out = c(err1, err2)
		return(out)
	}

	x = sapply((1:1000), FUN=Fcn)

	hist(x[ , 1], col=ColAlpha('blue', 0.5), xlim=c(0.03, 0.07))
	hist(x[ , 2], add=TRUE, col=ColAlpha('red', 0.5))

}

### given rating data, this function calculates different consistency metrics
GetConsistencyMetrics = function(
	  data, resp='resp', unitCol='unit', raterCol='rater') {

	formulaText = paste0(resp, '~', unitCol)
	fit = aov(formula = as.formula(formulaText), data=data)
	summary(fit)

	mod = lm(formula = as.formula(formulaText), data=data)
	summary(mod)[["adj.r.squared"]]
	summary(mod)[["r.squared"]]
	summary(mod)[["fstatistic"]]
	summary(mod)[["sigma"]]

	AvgFcn = mean
	DeltaFcn = function(x, y){(x-y)^2}
	ssTot = sum(DeltaFcn(data[ , resp], AvgFcn(data[ , resp])))
	ssRes = sum(summary(mod)[["residuals"]]^2)

	treatMeans = aggregate(
		  formula=as.formula(formulaText),
		  data=data[ , c(unitCol, resp)],
		  FUN=mean)[['resp']]

	withinVars = aggregate(
		  formula=as.formula(formulaText),
		  data=data[ , c(unitCol, resp)],
		  FUN=var)[['resp']]

	betweenVar = var(treatMeans)
	meanWithinVar = mean(withinVars, na.rm=TRUE)


	## furball:
	# Details:
	#   This is the ratio of the variance of the groupwise means divided by
	#   the mean of the groupwise variances where the grouping is defined
	#   by the ids variable
	# Function name on furball: BwRatioCalculator
	bwRatio = betweenVar / meanWithinVar

	ssTreat = sum(DeltaFcn(predict(mod), AvgFcn(data[ , resp])))
	ssTot
	ssTreat + ssRes
	p = sum(!is.na(mod[['coefficients']]))
	bigN = dim(data)[1]
	## degrees of freedom total and residual
	dfTot = bigN - 1
	dfErr = bigN - p
	## degrees of freedom group/treatment
	dfMod = p - 1
	fStats = (ssTreat/dfMod) / (ssRes/dfErr)
	fStats
	summary(mod)[["fstatistic"]]
	ssRes/dfErr
	rSq = 1 - ssRes/ssTot
	adjRsq = 1 - (ssRes/dfErr)/(ssTot/dfTot)
	#rSq
	#adjRsq
	# sum(DeltaFcn(treatMeans, AvgFcn(treatMeans)))
	#ssTreat + ssRes
	#ssTot

	raterPerUnit = table(data[ , unitCol])
	betweenPairsNum = sum(raterPerUnit*(raterPerUnit-1))
	allPairsNum = bigN*(bigN-1)

	consData = CreateConsData(
		  data, unitCol='unit', raterCol='rater', resp='resp')
	#dim(consData)

	consDataIrr = Create_consDataIrr(
		  data=data, unitCol=unitCol, raterCol=raterCol, resp=resp)

	irr = irr::kripp.alpha(t(as.matrix(consDataIrr)), method='interval')[['value']]

	res = CalcRaterCons(data, unitCol='unit', raterCol='rater', resp='resp',
		DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean)

	## so lets compare pwDelta with:
	#2*ssTot/(bigN-1)

	outList = list('bigN'=bigN,
		'rSq'=rSq,
		'adjRsq'=adjRsq,
		'fStats'=fStats,
		'dfMod'=dfMod,
		'dfErr'=dfErr,
		'dfTot'=dfTot,
		'bwRatio'=bwRatio,
		'irr'=irr
		)
	outList = c(outList, res)
	return(outList)
}

### this function simulates ratings and calculates consistency metrics
SimulateMultipleCalcMetrics = function(
	sampleNum=200,
	unitNum=200,
	raterNum=30,
	raterMeanSampler=function(n)runif(n, -1, 1),
	unitMeanSampler=function(n)runif(n, 0, 1),
	errDist=function(x)rnorm(1, mean=x),
	raterNumRange=c(2, 3, 4)) {

	data = SimulRatings(
		  unitNum=unitNum, raterNum=raterNum, unitMeanSampler=unitMeanSampler)
	value = GetConsistencyMetrics(
		  data, resp='resp', unitCol='unit', raterCol='rater')

	out = matrix(NA, sampleNum, length(value))
	for (i in 1:sampleNum)
	{
		data = SimulRatings(
			  unitNum=unitNum, raterNum=raterNum, unitMeanSampler=unitMeanSampler)
		value = GetConsistencyMetrics(data, resp='resp', unitCol='unit')
		out[i, ] = unlist(value)
	}
	out = as.data.frame(out)
	names(out) = names(value)
	return(out)
}


Example = function() {

	data = SimulateMultipleCalcMetrics(
			sampleNum=20, unitNum=200, raterNum=30,
			raterMeanSampler=function(n)runif(n, -1, 1),
			unitMeanSampler=function(n)runif(n, 0, 1),
			errDist=function(x)rnorm(1, mean=x),
			raterNumRange=c(2, 3, 4))

	### calculate confidence significance levels
	data = SimulateMultipleCalcMetrics(
			sampleNum=500, unitNum=200, raterNum=30,
			raterMeanSampler=function(n)runif(n, -1, 1),
			unitMeanSampler=function(n)runif(n, 0, 0),
			errDist=function(x)rnorm(1, mean=x),
			raterNumRange=c(2, 3, 4))

	metrics = c("rSq", "adjRsq", "fStats", "bwRatio", "irr", "consCoef")
}

## builds a density function from a vector x
CreateDensityFromVec = function(x) {
	Den = approxfun(density(x))
	xMin = min(x)
	xMax = max(x)
	d = xMax - xMin
	xMin = xMin - d/3
	xMax = xMax + d/3
	grid = seq(xMin, xMax, d/100)
	denX = Den(grid)
	return(list("Den"=Den, "grid"=grid, "denX"=denX))
}


ExampleSimulationStudy = function() {


	PlotDfList = function(
		dfList,
		legendLabels='',
		metrics=c("rSq", "adjRsq", "fStats", "bwRatio", "irr", "consCoef")) {

		DenList = list()
		gridList = list()
		xList = list()
		denXlist = list()

		m = matrix(c(1, 2, 3, 4, 5, 6, 7, 7, 7), nrow=3, ncol=3, byrow=TRUE)
		layout(mat=m, heights=c(0.4, 0.4, 0.2))

		#par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE)
		#par(mfrow=c(2, 3))
		for (j in 1:length(metrics)) {
			par(mar = c(2, 2, 1, 1))
			metric = metrics[j]
			grid = NULL

			for (i in 1:length(dfList)) {
				xList[[i]] =  dfList[[i]][ , metric]
				res = CreateDensityFromVec(xList[[i]])
				gridList[[i]] = res[['grid']]
				DenList[[i]] = res[['Den']]
				grid = c(grid, gridList[[i]])
			}

			grid = sort(grid)

			yMax = 0
			for (i in 1:length(dfList)) {
				denXlist[[i]] = DenList[[i]](grid)
				yMax = max(c(yMax, denXlist[[i]]), na.rm=TRUE)
			}


			plot(
				  grid, denXlist[[1]], main=metric, col=ColAlpha(1, 0.5),
				  ylim=c(0, yMax), type='l', ylab='density', xlab='x')
			for (i in 2:length(dfList)) {
				lines(grid, denXlist[[i]], col=ColAlpha(i, 0.5))
			}
		}
		plot(1, type="n", axes=FALSE, xlab="", ylab="")
		legend(
			  x="top", inset=0, legend=legendLabels,
		    col=ColAlpha(1:length(dfList), 0.5),
		    lwd=5, cex=0.8, horiz=TRUE)
	}



	dfList = list()

	dfList[[1]] = SimulateMultipleCalcMetrics(
		  sampleNum=100,  unitNum=100, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList[[2]] = SimulateMultipleCalcMetrics(
		  sampleNum=100,  unitNum=100, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(6, 7))

	dfList[[3]] = SimulateMultipleCalcMetrics(
		  sampleNum=100,  unitNum=100, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=(1:10))

	dfList[[4]] = SimulateMultipleCalcMetrics(
		  sampleNum=100,  unitNum=100, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=(1:20))

	dfList1 = dfList
	legendLabels1 = c(
		  '# rater: 2-3', '# raters 6-7', '# raters 1-10', '# raters 1:20')
	PlotDfList(dfList=dfList1, legendLabels=legendLabels1)

	## change unit number: 20 to 400
	dfList = list()
	dfList[[1]] = SimulateMultipleCalcMetrics(
		  sampleNum=100, unitNum=10, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList[[2]] = SimulateMultipleCalcMetrics(
		  sampleNum=100, unitNum=20, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList[[3]] = SimulateMultipleCalcMetrics(
		  sampleNum=100, unitNum=50, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList[[4]] = SimulateMultipleCalcMetrics(
		  sampleNum=100, unitNum=100, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList[[5]] = SimulateMultipleCalcMetrics(
		  sampleNum=100, unitNum=400, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList2 = dfList
	legendLabels2 = c('10 units', '20 units', '50 units', '100 units', '400 units')
	PlotDfList(dfList2, legendLabels=legendLabels2)

	## change distribution of the err
	dfList = list()
	dfList[[1]] = SimulateMultipleCalcMetrics(
		  sampleNum=200, unitNum=200, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rnorm(1, mean=x),
		  raterNumRange=c(2, 3))

	dfList[[2]] = SimulateMultipleCalcMetrics(
		  sampleNum=200, unitNum=200, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rbeta(1, shape1=x, shape2=1),
		  raterNumRange=c(2, 3))

	dfList[[3]] = SimulateMultipleCalcMetrics(
		  sampleNum=200, unitNum=200, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rgamma(1, shape=x, rate=1),
		  raterNumRange=c(2, 3))

	dfList[[4]] = SimulateMultipleCalcMetrics(
		  sampleNum=200, unitNum=200, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rgamma(1, shape=x, rate=5),
		  raterNumRange=c(2, 3))

	dfList[[5]] = SimulateMultipleCalcMetrics(
		  sampleNum=200,  unitNum=200, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)rgamma(1, shape=1, rate=x),
		  raterNumRange=c(2, 3))

	dfList[[6]] = SimulateMultipleCalcMetrics(
		  sampleNum=200, unitNum=200, raterNum=30,
		  unitMeanSampler=function(n){0*runif(n, 0, 1)},
		  errDist=function(x)runif(1, min=0, max=exp(x)),
		  raterNumRange=c(2, 3))


	dfList3 = dfList
	legendLabels3 = c(
		  'err: norm(x,1)', 'err: beta(1,x)', 'err: gamma(1,x)', 'err: gamma(5,x)',
		  'err: gamma(x,1)', 'err: unif(0, x)')

	PlotDfList(dfList3, legendLabels=legendLabels3)

	PlotDfList(dfList1, legendLabels=legendLabels1)
	PlotDfList(dfList2, legendLabels=legendLabels2)
	PlotDfList(dfList3, legendLabels=legendLabels3)


	##   $\Sum_i^n \Sum_j^n (x_i - x_j )^2 (i \neq j)   =    2*n \Sum_i (x_i - \bar{x})^2$
	##   Therefore:  $pwDelta = \Sum_i \Sum_j (x_i - x_j )^2 (i \neq j)/(n(n-1)) = 2*n/(n*(n-1)) \Sum_i (x_i - \bar{x})^2 $
	##  = 2/(n-1) \Sum_i (x_i - \bar{x})^2


	## \Sum_u \Sum_{r \neq s} (x(u,r) - x(u,s))^2 = \Sum_u

}

#### Functions
## just for plotting, creating transparent colors
ColAlpha = function(colors, alpha=1.0) {
  r = col2rgb(colors, alpha=TRUE)
  r[4,] = alpha*255
  r = r/255.0
  return(rgb(r[1,], r[2,], r[3,], r[4,]))
}

## this applies to two vectors and considers all the cartesian product
PairWiseCalcAvg = function(x, y, DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean,
	sampleSizeLimit=NULL)  {

  # creates all possible pairs and then takes an average (AvgFcn)
  # of delta (DeltaFcn) for the pairs
  n = length(x)
  m = length(y)

  if (!is.null(sampleSizeLimit) && sampleSizeLimit < n) {
  	ind = sample(1:n, sampleSizeLimit, replace=FALSE)
  	x = x[ind]
  	n = sampleSizeLimit
  }

  if (!is.null(sampleSizeLimit) && sampleSizeLimit < m) {
  	ind = sample(1:m, sampleSizeLimit, replace=FALSE)
  	y = y[ind]
  	m = sampleSizeLimit
  }

  grid = expand.grid((1:n), (1:m))

  Fcn2 = function(g) {
  	DeltaFcn(x[g[1]], y[g[2]])
  }
  res = apply(grid, 1, Fcn2)
  out = AvgFcn(res)
  return(out)
}


## this considers all the pairs from one
PairWiseCalcAvgWithin = function(
	  x, DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean, sampleSizeLimit=NULL) {

  # creates all possible pairs and then takes an average (AvgFcn)
  # of delta (DeltaFcn) for the pairs
  n = length(x)
  if (!is.null(sampleSizeLimit) && sampleSizeLimit < n) {
  	ind = sample(1:n, sampleSizeLimit, replace=FALSE)
  	x = x[ind]
  	n = sampleSizeLimit
  }

  Fcn = function(pair) {
  	DeltaFcn(pair[1], pair[2])
  }

  res = combn(x, 2, FUN=Fcn)
  out = AvgFcn(res)
  return(out)
}

Example = function() {
	n = 150
	m = 100
	x = rnorm(n)
	y = 2*x[1:m] + rnorm(m)
	PairWiseCalcAvg(x, y)
	PairWiseCalcAvg(x, x, sampleSizeLimit=200)
	PairWiseCalcAvgWithin(x)
}


## x, y are matched columns, each row is a pair of ratings for the same item
RaterConsCoef = function(
	  x, y, allRatings=NULL, DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean,
	  sampleSizeLimit=NULL) {

	# Krip Alpha using x, y
	# x[i] and y[i] are different raters ratings for the same unit
	delta = AvgFcn(DeltaFcn(x, y))
	if (is.null(allRatings)) {
		pwDelta = PairWiseCalcAvg(x=x, y=y,
	                DeltaFcn=DeltaFcn,
	                AvgFcn=mean,
	                sampleSizeLimit=sampleSizeLimit)
	} else {
		pwDelta = PairWiseCalcAvgWithin(x=allRatings,
	                DeltaFcn=DeltaFcn,
	                AvgFcn=mean,
	                sampleSizeLimit=sampleSizeLimit)
	}
	return(list(delta=delta, pwDelta=pwDelta, consCoef=(1-delta/pwDelta)))
}


Example = function() {

	DeltaFcn = function(z){
		(z[1] - z[2])^2
	}

	AvgFcn = mean
	delta = AvgFcn((x-y)^2)

	res = NULL
	for (i in 1:1000) {
		x = rnorm(100)
		y = rnorm(100)
		res = c(res, RaterConsCoef(x, y, DeltaFcn, AvgFcn)[[3]])
	}

	PlotFcn = function(res) {
		hist(res, main=mainText, col='blue')
		abline(v=quantile(res, 0.99), col='red', lwd=2)
		text(0.3, 150, '99th percentile')
	}


	mainText = 'Hist of Consistency for normal random ratings'
	png(paste(odir, gsub(" ", "_", mainText), '.png', sep=''))
	PlotFcn(res)
	dev.off()

    ### the case from uniform 0, 4 with 0.5 increments
  for (i in 1:1000) {
		x = sample(seq(0, 4, 0.5), 100, replace=TRUE)
		y = sample(seq(0, 4, 0.5), 100, replace=TRUE)
		res = c(res, RaterConsCoef(x, y, DeltaFcn, AvgFcn)[[3]])
	}

	mainText = 'Hist of Consistency for unif discrete random ratings'
	png(paste(odir, gsub(" ", "_", mainText), '.png', sep=''))
	PlotFcn(res)
	dev.off()


	z = data[ , resps[1]]
	z = na.omit(z)
	### the case from uniform 0, 4 with 0.5 increments
  for (i in 1:1000) {
		x = sample(seq(0, 4, 0.5), 100, replace=TRUE)
		y = sample(seq(0, 4, 0.5), 100, replace=TRUE)
		res = c(res, RaterConsCoef(x, y, DeltaFcn, AvgFcn)[[3]])
	}

	mainText = 'Hist of Consistency for random ratings from Utility data'
	png(paste(odir, gsub(" ", "_", mainText), '.png', sep=''))
	PlotFcn(res)
	dev.off()


	z = data[ , resps[2]]
	z = na.omit(z)
	### the case from uniform 0, 4 with 0.5 increments
  for (i in 1:1000) {
		x = sample(seq(0, 4, 0.5), 100, replace=TRUE)
		y = sample(seq(0, 4, 0.5), 100, replace=TRUE)
		res = c(res, RaterConsCoef(x, y, DeltaFcn, AvgFcn)[[3]])
	}

	mainText = 'Hist of Consistency for random ratings from Speech Quality data'
	png(paste(odir, gsub(" ", "_", mainText), '.png', sep=''))
	PlotFcn(res)
	dev.off()
}

CreateConsData = function(data, unitCol, raterCol, resp) {

	# data has the columns: raterCol, unitCol, resp
	# We output a data.frame with x,y where
	# x[i] and y[i] are different ratings for same unit
	outData = NULL
	if (sum(c(unitCol, raterCol, resp) %in% names(data)) == 3) {
		dat2 = data[ , c(unitCol, raterCol, resp)]
		dat2 = na.omit(dat2)
		dat3 = merge(dat2, dat2, by=c(unitCol), all=TRUE)
		dat4 = dat3[
		    dat3[ , paste(raterCol, '.x', sep='')] !=
		    dat3[ , paste(raterCol, '.y', sep='')], ]

		x = dat4[ ,  paste(resp, '.x', sep='')]
		y = dat4[ ,  paste(resp, '.y', sep='')]
		outData = data.frame(x=x, y=y)
	}
	return(outData)
}


Example = function() {
	data = SimulRatings(unitNum=100, raterNum=30,
		unitMeanSampler=function(n){1.2*runif(n, 0, 1)})
	consData = CreateConsData(data, unitCol='unit', raterCol='rater', resp='resp')
	dim(data)
	dim(consData)
	PairWiseCalcAvg(consData[ , 'x'], consData[ , 'y'])

	PairWiseCalcAvgWithin(consData[ , 'x'])
	PairWiseCalcAvgWithin(consData[ , 'y'])
	PairWiseCalcAvgWithin(data[ , resp])


	RaterConsCoef(x=consData[ , 'x'], y=consData[ , 'y'], allRatings=NULL)
	RaterConsCoef(
		  x=consData[ , 'x'], y=consData[ , 'y'], allRatings=data[ , resp])
}


CreateConsData_multipleAvg  = function(
	  data, unitCol, raterCol, resp, raterNum, AvgFcn=mean) {

	# data has the columns: raterCol, unitCol, resp
	# We output a data.frame with x,y where
	# x[i] avg of raterNum different raters
	# y[i] avg of raterNum different raters and disjoint from x[i]
	data = data[ , c(unitCol, raterCol, resp)]
	data = na.omit(data)

	Fcn = function(unit) {
		unitData = data[data[ , unitCol]==unit, ,drop=FALSE]
		k = dim(unitData)[1]
		if (k < 2*raterNum) {
			return(NULL)
		}
		# first we get the subsets of size 2*raterNum
		sets = combn(1:k, 2*raterNum, simplify=TRUE)
		# then we need to further divide these subsets to two parts
		# there is multiple ways to do that
		setsParts = combn(1:(2*raterNum), raterNum, simplify=TRUE)
		# we only need half of these sets
		# setsParts = setsParts[ , 1:(dim(setsParts)[2]/2)]
		setsNum = dim(sets)[2]
		setsPartsNum = dim(setsParts)[2]
		outDf = data.frame('x'=NA, 'y'=NA)

		for (i in 1:setsNum) {
			part = sets[ , i]
			print(part)
			for (j in 1:(setsPartsNum/2)) {
				print(i); print(j)
				part1 = setsParts[ , j]
				part2 = setsParts[ , (setsPartsNum-j+1)]
				print(part1); print(part2)
				part1 = part[part1]
				part2 = part[part2]
				# print(part1); print(part2)
				part1 = unitData[ , resp][part1]
				part2 = unitData[ , resp][part2]
				ratingPair = c(AvgFcn(part1), AvgFcn(part2))
				print(ratingPair)
				outDf = rbind(outDf, ratingPair)
			}

		}
		outDf = outDf[-1, ,drop=FALSE]
		return(outDf)
	}

	unitSet = unique(data[ , unitCol])
	dfList = lapply(X=unitSet, FUN=Fcn)
	df = do.call("rbind", dfList)
	return(df)
}




CalcRaterCons = function(
	  data, unitCol, raterCol, resp,
	  DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean) {
	# calculates Kripps alpha given data
	consData = CreateConsData(data, unitCol=unitCol, raterCol=raterCol, resp=resp)
	out = RaterConsCoef(
		  x=consData[ , 1], y=consData[ , 2], allRatings=data[ , resp],
		  DeltaFcn=DeltaFcn, AvgFcn=AvgFcn)
	return(out)
}

BootStrapAlphaConsData = function(
	  consData, DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean,
	  bsNum=100, bsSampleSize=NULL) {

	# given data with columns unitCol, raterCol, resp
	# it calculates multiple delta, pwDelta, and kripp alpha bsNum time
	# it limits the sample size to bsSampleSize
	if (is.null(bsSampleSize)) {
		bsSampleSize = dim(consData)[1]
	}
	bsSampleSize = min(c(dim(consData)[1], bsSampleSize))


	consDf = data.frame(delta=NA, pwDelta=NA, consCoef=NA)
	for (i in 1:bsNum) {
		## too big so we subsample
		x = consData[ , 1]
		y = consData[ , 2]
		ind = sample(1:length(x), bsSampleSize, replace=TRUE)
		out = RaterConsCoef(x=x[ind], y=y[ind], DeltaFcn=DeltaFcn, AvgFcn=AvgFcn)
		out = unlist(out)
		consDf = rbind(consDf, out)
		print(i)
	}
	return(consDf)
}

BootStrapAlpha = function(
	  data, unitCol, raterCol, resp,
    DeltaFcn=function(x, y){(x-y)^2}, AvgFcn=mean,
	  bsNum=100, bsSampleSize=NULL) {

	# given data with columns unitCol, raterCol, resp
	# it calculates multiple delta, pwDelta, and kripp alpha bsNum times
	consData = CreateConsData(data, unitCol=unitCol, raterCol=raterCol, resp=resp)
	out = BootStrapAlphaConsData(consData=consData, DeltaFcn=DeltaFcn, AvgFcn=AvgFcn,
		bsNum=bsNum, bsSampleSize=bsSampleSize)
	return(out)
}


PlotScat = function(consDf, odir, mainText, save=FALSE) {

	if (save) {
		fn = paste(odir, gsub(" ", "_", mainText), '.png', sep='')
		Cairo(file=fn, type='png')
		print(fn)
	}

	maxValue = max(c(consDf[ , 'delta'], consDf[ , 'pwDelta']), na.rm=TRUE)

	plot(
		  consDf[ , 'delta'], consDf[ , 'pwDelta'], col=ColAlpha('blue', 0.5),
		  pch=20, cex=2,
		  xlab='inter-item avg delta', ylab='total avg delta', main=mainText,
		  cex.lab=1.5, cex.main=1.5, cex.axis=1.2, xlim=c(0, maxValue),
		  ylim=c(0, maxValue))

	abline(0, 1, col=ColAlpha('black', 0.75), lty=2)
	grid (NULL,NULL, lty = 6, col = "grey")

	if (save) {dev.off()}
}

PlotHist = function(consDf, odir, mainText, save=FALSE) {

	if (save) {
		fn = paste(odir, gsub(" ", "_", mainText), '.png', sep='')
		Cairo(file=fn, type='png')
		print(fn)
	}

	hist(
      consDf[ , 'consCoef'], col='orange', xlab='rater consist coef',
      main=mainText, cex.lab=1.5, cex.main=1.5, cex.axis=1.2, xlim=c(-0.5, 1.1))
	if (save) {
		dev.off()
	}
}


PlotHistScat = function(
	  consDf, resp, respName, odir, extraText='', save=FALSE) {

	mainText = paste(extraText, 'delta vs pwdelta in for ', respName, sep='')
	print(mainText)
	PlotScat(consDf, odir=odir, mainText, save=save)
	mainText=paste(extraText, 'Hist of rater cons in for ', respName, sep='')
	PlotHist(consDf, odir=odir, mainText, save=save)
	print(mainText)
}



#### PART II
# This is a function to calculate Util
# u is a vector of values between 0 and 1
# missing is replaced by 0.2/4
Util = function(u, at, naReplace=(0.2/4)) {

	if (length(u) < at) {
      v = rep(naReplace, (at-length(u)))
      u = c(u, v)
  }
   u2 = u^2
   w = 1 / (1:at)
   out = sqrt(sum(u2*w) / sum(w))
   return(out)
}

Util5 = function(u) {
	Util(u, at=5)
}

Util10 = function(u) {
	Util(u, at=10)
}

Example = function()  {
	u = c(1, 1, 1)
	at = 5
	Util(u, at)
}

## Calculate maximum of the diff
# the maximum possible difference
# between Util5 and Util10
Example = function() {

	UtilDiff = function(x, y, b=sum(1/(1:5)), d=sum(1/(6:10))) {
		abs(sqrt(x / b) - sqrt((x + y) / (b + d)))
	}

	b = sum(1/(1:5))
	d = sum(1/(6:10))

	MakeImagePlot = function(b0, d0, text) {
		n = 1000
		xVec = seq(0, b0, b0/n)
		yVec = seq(0, d0, d0/n)
		xyGrid = expand.grid(xVec,yVec)

		gGrid = UtilDiff(xyGrid[ , 1], xyGrid[ , 2])
		m1 = max(gGrid)
		m2 = min(gGrid)

		mainText = paste(text, ', ', 'max=', signif(m1, 2), sep='')

		gMat = matrix(gGrid, length(xVec), length(yVec))
		image(
		  	xVec, yVec, gMat, col=topo.colors(100, alpha=1), main=mainText,
			  xlim=c(-0.1, b+0.1), ylim=c(-0.1, d+0.1), zlim=c(0, 0.5),
			  xlab='x', ylab='y')
		contour(xVec, yVec, gMat, add=TRUE, col="black", labcex=1, lwd=2)
		return(m1)
	}

	b1 = sum((1/(1:5))*c(1, (3/4)^2, (3/4)^2, (3/4)^2, (3/4)^2))
	d1 = sum((1/(6:10))*c(1, (3/4)^2, (3/4)^2, (3/4)^2, (3/4)^2))


	library('Cairo')

	fn = paste(ofigs, 'max_diff.png', sep='')
	Cairo(file=fn, type='png')
	#png(fn)
	par(mfrow=c(1, 2), pty="s")
	MakeImagePlot(b, d, text='stdrd Util')
	MakeImagePlot(b1, d1, text='non-strd Util')
	dev.off()

	fn1 = paste(ofigs,'max_diff1.png',sep='')
	fn2 = paste(ofigs,'max_diff2.png',sep='')
	#Cairo(file=fn,type='png')
	#png(fn)

	Cairo(file=fn1, type='png')
	MakeImagePlot(b, d, text='stdrd Util')
	dev.off()

	Cairo(file=fn2, type='png')
	MakeImagePlot(b1, d1, text='non-strd Util')
	dev.off()

	rm(fn1)
	rm(fn2)
}
