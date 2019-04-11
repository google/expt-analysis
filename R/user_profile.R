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

# source functions for analyzing device/user profiles
# for generic timestamped usage data

## this function splits a time interval
# specified by timeColStart, timeColEnd
# whenever a break point happens in the middle
# e.g. when we want to break by each hour start
# we assume the time columns are already in as.POSIXct type
# with the format: "2019-01-27 20:55:59"
SplitUsageTime_atBreakPoints = function(
    dt, timeColStart, timeColEnd,
    timeRounding="hours", durCol=NULL) {

  timeRoundingDict = list(
      "seconds"=1, "mins"=60, "hours"=3600, "days"=3600*24)

  ## we keep track of unchanged rows vs changed rows
  # where time was partitioned
  dt[ , "time_split"] = FALSE
  print(dim(dt))
  # we observe if there are any points which needs breaking
  iterationNum = 0
  while (iterationNum == 0 || sum(ind) > 0) {

    dt[ , "next_break"] = (
      floor_date(
        dt[, get(timeColStart)], unit=timeRounding,
        week_start=getOption("lubridate.week.start", 7)) +
      timeRoundingDict[[timeRounding]])

    dt[ , c("ts", "next_break")]

    ind = dt[ , get(timeColEnd)] > dt[ , next_break]

    if (sum(ind) > 0) {
      # we create two data sets here instead of one
      dt1 = copy(dt[ind, ])
      dt2 = copy(dt[ind, ])
      # dt1 will have the first part of the time partition
      dt1[ , timeColEnd] = dt1[ , next_break]
      # dt2 will have the second partition
      dt2[ , timeColStart] = dt2[ , next_break]
      dt1[ , "time_split"] = TRUE
      dt2[ , "time_split"] = TRUE

      dt[ind, ] = dt1
      dt = rbind(dt, dt2)
      #dt = SortDf(dt, c(userCol, "ts"))
    }
    iterationNum = iterationNum + 1
    print(iterationNum)
  }

  print(dim(dt))

  if (!is.null(durCol)) {
    dt[ , durCol] = dt[ , get(timeColStart)] - dt[ , get(timeColEnd)]
  }

  return(dt)
}

TestSplitUsageTime_atBreakPoints = function() {

  # testing SplitUsageTime_atBreakPoints
  dt0 = data.table(data.frame(
    user_id=c("1", "1"),
    ts=c("2019-01-27 18:17:59", "2019-01-27 20:55:59"),
    end_ts=c("2019-01-27 18:18:59", "2019-01-27 22:05:59"),
    usage=c("p1", "p2")))

  dt0[ , "ts"] = as.POSIXct(dt0[ , ts], tz="GMT")
  dt0[ , "end_ts"] = as.POSIXct(dt0[ , end_ts], tz="GMT")

  SplitUsageTime_atBreakPoints(
      dt=dt0, timeColStart="ts", timeColEnd="end_ts",
      timeRounding="hours", durCol=NULL)
}

## generic function which calculates a sleep time for each user/device/unit
# it finds top k largest timegaps between the end time (timeColEnd) of a usage
# and the start time of the next usage
# then it takes the median
GetSleepTime = function(dt, userCol, timeColStart, timeColEnd) {

  # we change system timezone to GMT
  # to avoid R changing timezones and causing issues by adding NA below
  # in this code: next_ts=c(ts[-1], NA)
  Sys.setenv(TZ="GMT")
  dt2 = SortDf(dt, cols=c(userCol, timeColStart))
  # finding the next start time and adding it to each row
  dt2 = dt2[ , `:=`(next_ts=c(get(timeColStart)[-1], NA)), by=get(userCol)]
  # calculate time_gap
  dt2[ , "time_gap"] = dt2[ , next_ts] - dt2[ , get(timeColEnd)]

  # time_gap should be non-negative if data is consistent
  negGapPercent = dim(dt2[time_gap < 0, ])[1] / dim(dt2)[1]

  timeGaps = as.numeric(dt2[ , time_gap])

  pltList = list()

  PltGaps = function() {

    DichomHist(timeGaps, step=0.05, labelCol="red")

  }

  dt3 = dt2[ , mget(c(userCol, "time_gap"))]

  dt3[ , "time_gap"] = as.numeric(dt3[ , time_gap])

  ## we take the median of the top longest time_gaps
  dtSleepTime = dt3[ ,
      .(sleep_time=median(sort(time_gap, decreasing=TRUE)[1:6]/3600)),
      by=get(userCol)]

  colnames(dtSleepTime) = c(userCol, "sleep_time")

  sleepTimes = dtSleepTime[ , sleep_time]

  PltSleepTime = function() {
    DichomHist(x=sleepTimes, step=0.05, labelCol="red", srt=90)
  }

  return(list(
      "dt"=dtSleepTime,
      "PltGaps"=PltGaps,
      "PltSleepTime"=PltSleepTime,
      "negGapPercent"=negGapPercent))
}

## it is possible sleep hours are not consecutive
# we make them consecutive
Continuous_sleepHours = function(sleepHours, numSleepHours) {
  hours = 0:23
  sleepHours_midPoint = round(mean(sleepHours))
  distVec = abs(hours - sleepHours_midPoint)
  thresh = sort(distVec)[numSleepHours]
  ind = which(distVec <= thresh)
  sleepHours_continuous = hours[ind][1:numSleepHours]
  return(sleepHours_continuous)
}

TestContinuous_sleepHours = function() {

  Continuous_sleepHours(
    sleepHours=1:6, numSleepHours=6)
  Continuous_sleepHours(
    sleepHours=c(1, 3, 4, 5, 8, 10), numSleepHours=6)
}

## this function processes raw data and returns a few data frames
# dt: a processed table which is similar to raw
# with added columns eg hour, date
# dtSplit: similar to dt but it splits the time interval to two\
# whenever a time interval
# crosses an hour
# dtDailyTotals: includes daily total metrics
ProcessRawData = function(
    dt,
    userCol,
    usageCol,
    durCol,
    localTimeCol,
    savePlts=FALSE) {

  ## require timestamp in this format: "2019-01-21 21:01:57"
  # ts stands for timestamp
  dt[ , "ts"] = as.POSIXct(dt[ , get(localTimeCol)], tz="GMT")
  dt[ , durCol] = floor(dt[ , get(durCol)])

  ## extract hour of day (hour)
  dt[ , "hour"] = hour(dt[ , get(timeCol)])
  dt[ , "date"] = date(dt[ , get(timeCol)])
  dt[ , "dow_string"] = weekdays(dt[ , get(timeCol)])
  dt[ , "dow"] = as.POSIXlt(dt[ , date])[["wday"]]
  dt[ , userCol] = as.character(dt[ , get(userCol)] )
  cols = c(
      userCol,"ts", "date", "dow", "dow_string", "hour", durCol,
      usageCol, sliceCols)
  dt = dt[ , mget(cols)]

  x = dt[ , get(durCol)]
  # hist(x)
  # summary(x)
  # DichomHist(x, step=0.05, labelCol="red")

  ## count number of units, number of usage possibilities, date range
  unitNum = length(unique(dt[ , get(userCol)]))
  print("number of unique users in data:")
  print(unitNum)

  dates = sort(unique(dt[ , date]))
  dateNum = length(dates)
  print("number of unique dates in data:")
  print(dateNum)

  ## we might need to get rid of edge dates
  # so we assure all dates have complete data
  dropEdgeDates = TRUE

  if (dropEdgeDates) {
    dates2 = dates[!dates %in% c(max(dates), min(dates))]
    if (length(dates2) == 0) {
      warning("no date is remaining")
    }
    dt = dt[date %in% dates2, ]
  }

  print("number of unique dates in data after removing boundary dates:")
  dates = sort(unique(dt[ , date]))
  dateNum = length(dates)
  print(dateNum)

  userNum = length(unique(dt[ , get(userCol)]))
  print("number of users in data:")
  print(userNum)
  dates = sort(unique(dt[ , date]))
  dateNum = length(dates)

  dt[ , "end_ts"] = dt[ , ts] + dt[ , get(durCol)]
  # dt[ , mget("ts", "end_ts", durCol)]

  timeRounding = "hours"

  dtSplit = SplitUsageTime_atBreakPoints(
      dt=dt, timeColStart="ts", timeColEnd="end_ts",
      timeRounding="hours", durCol=durCol)

  cols = c(userCol, usageCol, "ts", "end_ts", "time_split")

  #print(dim(dtSplit))
  dtSplit = SortDf(dtSplit, c(userCol, "ts"))

  ## check if the splitting has worked
  dtSplit[time_split == TRUE, mget(cols)]

  ## re-assign hour
  dtSplit[ , "hour"] = hour(dtSplit[ , get(timeCol)])

  ## aggregate
  unitCols = c(userCol, "date")

  dtDailyTotals = dt[ , .(
      total_dur=sum(get(durCol)),
      distinct_usage_num=length(unique(get(usageCol))),
      usage_num=.N), by=unitCols]

  SortDf(dtDailyTotals , cols=unitCols)
  dtNumActiveHours = dt[ , .(
      num_active_hours=length(unique(hour))), by=unitCols]

  # hist(dtNumActiveHours[ , num_active_hours], col="skyblue")

  dim(dtDailyTotals)
  dim(dtNumActiveHours)

  dtDailyTotals = merge(
      dtDailyTotals, dtNumActiveHours,
      by=unitCols, all.x=TRUE)

  ##
  timeColStart = "ts"
  timeColEnd = "end_ts"
  cols = c(userCol, timeColStart, timeColEnd, durCol)
  dt2 = copy(dt[ , mget(cols)])

  res = GetSleepTime(
      dt=dt2,
      userCol=userCol,
      timeColStart=timeColStart,
      timeColEnd=timeColEnd)

  print("percent of negative time gaps due to data inaccuracy:")
  print(res[["negGapPercent"]])
  dtSleepTime  = res[["dt"]]

  dtDailyTotals = merge(dtDailyTotals, dtSleepTime, by=userCol)

  aggCols = c(
      "total_dur",
      "distinct_usage_num",
      "usage_num",
      "num_active_hours",
      "sleep_time")

  if (savePlts) {
    for (col in aggCols) {

      fn = paste0(figsPath, col, "_hist.png")

      print(fn)

      Cairo(file=fn, type="png", bg=ColAlpha("white", 0.75))

      hist(
          dtDailyTotals[ , get(col)],
          probability=TRUE,
          col="skyblue",
          main=Capitalize(gsub("_", " ", col)),
          xlab=col)

      dev.off()

      fn = paste0(figsPath, col, "_dichom_hist.png")

      print(fn)

      Cairo(file=fn, type="png", bg=ColAlpha("white", 0.8))

      DichomHist(
          x=dtDailyTotals[ , get(col)],
          step=0.05,
          labelCol="red",
          srt=90)

      dev.off()

      #close(fn)
    }
  }

  return(list(
      "dt"=dt,
      "dtSplit"=dtSplit,
      "dtDailyTotals"=dtDailyTotals,
      "aggCols"=aggCols
      ))

}

### PART 2: distance methods

## calculate median distance
MedianDist = function(x, y) {

  x = na.omit(x)
  y = na.omit(y)
  m1 = median(x)
  m2 = median(y)

  medMax = max(m1, m2)
  medMin = min(m1, m2)

  delta1 = sum((x > medMin)*(x < medMax)) / length(x)
  delta2 = sum((y > medMin)*(y < medMax)) / length(y)
  delta = (delta1 + delta2) / 2

  return(delta)
}

## we define a quantile function which slightly differs from regular quantile function
# the advantage is the quantile value is always observed in real data
# note that the R definition and classic definition does not satisfy that
# there are two values for each value p
# see left and right quantile definitions in Reza Hosseini, PhD Thesis, 2010, UBC
# https://open.library.ubc.ca/cIRcle/collections/ubctheses/24/items/1.0070885
QuantileFcn = function(x, direction="left") {

  x = na.omit(x)
  x = sort(x)
  n = length(x)
  G = floor
  if (direction != "left") {
    G = ceiling
  }

  F = function(p) {

    ind = G(p*n)
    ind = pmax(ind, 1)
    ind = pmin(ind, n) # unnecessary if p =< 1
    return(x[ind])
  }

  return(F)
}


TestQuantileFcn = function() {

  x = sample(1:50, 20)

  LQ = QuantileFcn(x)
  RQ = QuantileFcn(x, "right")
  g = seq(0, 1, 0.01)

  lq = LQ(g)
  rq = RQ(g)
  q = quantile(x, g)

  plot(g, lq, col=ColAlpha("blue"), type="l", lwd=2)
  lines(g, rq, col=ColAlpha("red"), lwd=2)
  lines(g, q, col=ColAlpha("black"), lwd=2)

  legend("topleft", legend=c("left quantile", "right quantile", ""))

}


QuantileRangeFcn = function(p, eps=0.1) {

  function(x) {
    p1 = max(min(p - eps, 1), 0)
    p2 = max(min(p + eps, 1), 0)

    x1 = quantile(x, p1)
    x2 = quantile(x, p2)

    return(c(x1, x2))
  }

}


TestQuantileRange = function() {

  x = 1:100
  MedianR = QuantileRangeFcn(p=1/2, eps=0.1)
  MedianR(x)
}


#ProbabilityDist = function() {
#
#}


## In progress
MedianDist = function(x, y) {

  mr = MedianR(x)
  # get the central values of x
  ind = x %in% mr

  x1 = x[ind]
  y1 = y[ind]

}

TestMedianDist = function() {

  par(mfrow=c(2, 2))
  x = rnorm(100)
  y = rnorm(100)
  plot(x, y, col=ColAlpha("blue", 0.1), pch=10)
  MedianDist(x, y)

  x = runif(10000, -1, 1)
  y = sqrt(1 - x^2)
  plot(x, y, col=ColAlpha("blue", 0.1), pch=10)
  MedianDist(x, y)

  y = sqrt(1 - x^2) + rnorm(1000)
  plot(x, y, col=ColAlpha("blue", 0.1), pch=10)
  MedianDist(x, y)

}
