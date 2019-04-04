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



### plot the results
def DisplayResults(seqDfWithSignif):

  metrics = ['relative_prob_lower1', 'relative_prob_lower2']
  plt.scatter(seqDfWithSignif[metrics[0]], seqDfWithSignif[metrics[1]])
  #plt.xscale('log')
  #plt.yscale('log')
  m = max(seqDfWithSignif[metrics[0]].max(), seqDfWithSignif[metrics[1]].max())
  plt.plot([0, m], [0, m], 'k-', lw=2, alpha=0.4)
  plt.plot([0, m], [1, 1], 'k-', lw=2, alpha=0.3)
  plt.plot([1, 1], [0, m], 'k-', lw=2, alpha=0.3)
  plt.xlabel(metrics[0], fontsize=18)
  plt.ylabel(metrics[1], fontsize=16)

  df0 = seqDfWithSignif.sort_values(['relative_prob1'], ascending=[0])[[
      'sequence',
      #'event_1',
      #'event_2',
      #'event_3',
      #'event_1_count_agg',
      #'event_2_count_agg',
      #'event_3_count_agg',
      #'event_1_to_2_count_agg',
      #'event_2_to_3_count_agg',
      #'event_1_to_3_count_agg',
      'count',
      'relative_prob1',
      'relative_prob2',
      'relative_prob_lower1',
      'relative_prob_lower2']]
  Mark(df0[:10])
