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

codePath = ""
codePath = paste0(
    "/google/src/cloud/rezani/cp1/google3/experimental/users/rezani/",
    "codes/expt-analysis/R/")

source(paste0(codePath, "data_analysis.R"))
source(paste0(codePath, "clustering.R"))
source(paste0(codePath, "plotting.R"))
source(paste0(codePath, "market_basket_analysis.R"))

library("arules")
library("data.table")

n = 5
df = data.frame(
    "id"=rep("", n),
    "date"=rep("", n),
    "full_seq_basket"=rep("", n),
    stringsAsFactors=FALSE)

df[1, ] =  list("F_3qGuHsw",  "2019-02-23",  "a;b;c")
df[2, ] =  list("G2CXtn3zk",  "2019-02-02",  "c;d;e")
df[3, ] =  list("L7nkvoEik",  "2019-02-05",  "e;f;g")
df[4, ] =  list("LyebHtPg4",  "2019-02-06",  "f;d;h")
df[5, ] =  list("LyebHtPg4",  "2019-02-16",  "a;b;c;d")


res = Find_sigBaskets_fromSeqData(
    seqDf=df,
    idCols=c("id", "date"),
    basketCol="full_seq_basket",
    liftThresh=1.5,
    suppThresh=0.00001,
    confThresh=0.5,
    lhsSizeThresh=2,
    basketSizeLowerBound=1,
    keepFreqItemOnly=FALSE,
    topItemNum=200,
    writePath=NULL)
