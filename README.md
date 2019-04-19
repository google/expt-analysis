# Demo code for analyzing experiment / observational data
* R code includes code to analyze experiment data and implements
also a variance reduction method
* Python code includes code to create sequential (journey) data using
timestamped event data and find statistically significant patterns
* code has also other functionalities which we will add a guide to in future



## Disclaimer

This is not an official Google project. This is only demo code for methods to
analyze experiments data for generic use cases. No real data is used in the code. All the examples are completely hypothetical and using
fabricated data. Existence of this code does not imply any analysis as such was performed on real data.


## Using the R Code for Variance reduction

### Make sure your data satisfies these conditions:
* your data is aggregated: every row corresponds to a single user_id
* user_id is an integer (so that bucketing works for jack-knife method)
* you have an expt_id column with labels "treat" and "cont"

### Reported estimators (which come with CIs):
Three estimators are reported with corresponding CIs:
* raw: this is just the raw metric
* adjDiff_contDataOnly: this is adjusted metrics which only uses control data
* adjDiff_withExptData: this is adjusted metric which uses control and experiment data

### Methods for calculating estimator CIs
Two methods are used to calculate CIs:
* Jack Knife with buckets (you can pick the number of your buckets)
* Bootstrap (no bucketing and therefore slow)

### Currently supported metrics
* Metric_meanRatio: for a variable y, this calculates the change in mean of y across users on treatment vs control by calculating their ratio

* Metric_sumRatio: for a variable y, this calculates the change in the sum of y across users on treatment vs control by calculating their ratio

* Metric_meanMinus: for a variable y, this calculates the change in mean of y across users on treatment vs control by calculating their delta (treat - control)

* Metric_sumMinus: for a variable y, this calculates the change in sum of y across users on treatment vs control by calculating their delta (treat - control)

* More sophisticated metrics will be available later. For example metrics involving two value columns e.g. CTR(treatment) / CTR(control)

### Bonus helper functions:
* A function (FitPred_multi) to test how well the predictors do to predict the value columns

* A function (Remap_lowFreqCategs) to deal with categorical variables with rare labels, to make sure the models do not fail


### Example with simulated data


`codePath = "~/data-analysis/R/"`

`readPath = "~/data/"`

`writePath = "~/data/"`

`source(paste0(codePath, "data_analysis.R"))`

`source(paste0(codePath,"plotting.R"))`

`source(paste0(codePath,"unit_analysis.R"))`

`library("data.table")`

`library("rglib")`

`library("ggplot2")`

`library("scales")`

`library("data.table")`

`library("gridExtra")`

`library("grid")`

`library("xtable")`

`ver = "v0"`

`SimData_write(ver=ver, parallel=TRUE, userNum=10^4, dataPath=writePath)`

`simData = ReadSimData(ver, dataPath=readPath)`

`userDt_fromUsage_obs = simData[["userDt_fromUsage_obs"]]`

`p1 = Check_forUnbalance(dt=dt, predCols=c("gender"))`

`p2 = Check_forUnbalance(dt=dt, predCols=c("country"))`

`p3 = Check_forUnbalance(dt=dt, predCols=c("gender", "country"))`

`valueCols= c("imp_count", "obs_interact", "obs_amount")`

`predCols=c("gender", "country")`

`CommonMetric = Metric_meanRatio`

`Mod = GenModFcn(20)`

`dt[ , "bucket"] = Mod(as.numeric(dt[ , user_id]))`

`res = CalcMetricCis_withBuckets(
    dt=dt, valueCols=valueCols, predCols=predCols, CommonMetric=CommonMetric,
    ci_method="jk_bucket")`

`ciDf_jk  = res[['ciDf']]`

`ciDf_jk = StarCiDf(
    df=RoundDf(ciDf_jk, 4), upperCol="ci_upper", lowerCol="ci_lower",
    upperThresh=c(1, 1.5, 2), lowerThresh=c(1, 0.75, 0.5))`

`print(ciDf_jk)`



## Using the Python Code for sequential data analysis

The code will use timestamped event data to generate sequential data which
tracks the specified properties of the events and allows for many interesting
ways to slice the data. See sequential_data_example.py for an example
with explanations. Also try "help(BuildAndWriteSeqDf)" for more info.
We will also provide examples for finding significant sequences in the future.
