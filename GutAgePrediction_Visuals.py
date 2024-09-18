import qiime2
import skbio
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiime2.plugins import feature_table, sample_classifier, metadata, phylogeny, diversity, emperor
import gzip
import time
import os

#mae calculation
def mae(predictions, actual):
    return abs(predictions - actual).mean()


#initializing and passing in data
t = qiime2.Artifact.load('raw.nobloom.minfeat.mindepth.biom.qza')
r = feature_table.actions.rarefy(t, 1000)
rt = r.rarefied_table
md = qiime2.Metadata.load('10317_20231212-135355.txt')
#p_tree = qiime2.Artifact.load('raw.nobloom.minfeat.mindepth.tree.qza')


#creating dataframe and removing nulls
df = md.to_dataframe()
df['host_age_normalized_years'] = pd.to_numeric(df['host_age_normalized_years'], errors='coerce')
df = df[~df['host_age_normalized_years'].isnull()]
df = df[~df['host_subject_id'].duplicated()]


#running 5 fold cross validation on full dataset
md = qiime2.Metadata(df)
rt_reg = feature_table.actions.filter_samples(\
        table = rt, metadata = md, where = "[env_package] == 'human-gut'")\
        .filtered_table
md_age = md.get_column('host_age_normalized_years')
reg = sample_classifier.actions.regress_samples_ncv(\
    table = rt_reg , metadata = md_age, estimator='RandomForestRegressor',\
    cv = 5, n_jobs = 16, n_estimators = 500, random_state = 123)

pred = reg.predictions.view(pd.Series)
act = md_age.to_dataframe()['host_age_normalized_years']
mae_ncv = mae(pred, act)

scatter = sample_classifier.actions.scatterplot(predictions = reg.predictions, truth = md_age).visualization
scatter.save('mdl_1000iter_scatterplot.qzv')

#creating dataframe of all iterations and corresponding MAEs
mae_array = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob('shared_mae_results_1000iter_*.csv')])
mae_array = mae_array.set_index('iteration')['result']

#lowest 5% MAE values from iterations
threshold_index = int(len(mae_array) * 0.05)
bottom_5_indices = list(np.argsort(mae_array, threshold_index)[:threshold_index])
bottom_5_indices = list(map(str, bottom_5_indices))

with open('bottom5_MAE_iterations.txt', 'w') as f:
    f.write('Iteration #s of bottom 5% MAE:')
    f.write('\n'.join(bottom_5_indices))


#plotting our MAE distribution
fig, ax = plt.subplots()

ax.hist(mae_array, bins=20, edgecolor='black', color='red', alpha=0.5, label='mae_vals')

mean_mae_vals = np.mean(mae_array)
ax.axvline(mean_mae_vals, color='red', linestyle='dashed', linewidth=2)

ax.axvline(mae_ncv, color='blue', linestyle='dashed', linewidth=2)

ax.set_xlabel('MAE Values')
ax.set_ylabel('Frequency')
ax.legend()
ax.set_title('Histogram of MAE Values for "Healthy" Data (1000 iterations)')

plt.savefig('mae_distribution_1000iter.png')
plt.show()
