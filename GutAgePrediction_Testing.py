import qiime2
import skbio
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


#conditions for healthy data
cond1 = df['diabetes'] == 'I do not have this condition'
cond2 = pd.to_numeric(df['host_body_mass_index'], errors='coerce') >= 18.5
cond3 = pd.to_numeric(df['host_body_mass_index'], errors='coerce') <= 35
cond4 = df['migraine'] == 'I do not have this condition'
cond5 = df['antibiotic_history'] == 'I have not taken antibiotics in the past year.'
cond6 = df['cardiovascular_disease'] == 'I do not have this condition'


#running 1000 iterations and storing testing sets
df_healthy = df[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
output_folder = 'rf-classifier-age-testing'
os.makedirs(output_folder, exist_ok=True)

healthy_md = qiime2.Metadata(df_healthy)
rt_healthy = feature_table.actions.filter_samples(\
        table = rt, metadata = healthy_md, where = "[env_package] == 'human-gut'")\
        .filtered_table
healthy_age = healthy_md.get_column('host_age_normalized_years')
mae_dict = []

iteration_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

results = []
for i in range(iteration_index, iteration_index + 1000, 10):
    
    healthy_reg = sample_classifier.actions.regress_samples(\
    table = rt_healthy , metadata = healthy_age, test_size = 0.2, estimator='RandomForestRegressor',\
    n_jobs = 16, n_estimators = 500)
    pred = healthy_reg.predictions.view(pd.Series)
    act = healthy_age.to_dataframe()['host_age_normalized_years']

    mae_result = mae(pred, act)
    results.append((i, mae_result))
    
    file_path = os.path.join(output_folder, f'test_iteration_{i}.txt.gz')

    with gzip.open(file_path, 'wt', encoding='utf-8') as fp:
        act[pred.index].to_csv(fp)

results = pd.DataFrame(results, columns=['iteration', 'result'])

results.to_csv(f'shared_mae_results_1000iter_{iteration_index}.csv', sep=',', index=False, header=True)