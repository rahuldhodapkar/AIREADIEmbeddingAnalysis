#!/usr/bin/env python
#
# Calculate relationship between p1, p2, p3 and clinical values
#


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import os
import re

################################################################################
## Create Ouptut Scaffolding
################################################################################

os.makedirs('./calc/cgm/bergman_stats', exist_ok=True)
os.makedirs('./fig/cgm/bergman_stats', exist_ok=True)


def get_file_paths_os_walk(directory):
    """ Takes a base directory and returns all image paths. """
    file_paths = []
    file_extensions = ('.json') 
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(file_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths

paths = get_file_paths_os_walk('./data/aireadi/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6')

################################################################################
## Read Data
################################################################################



bergman_df = pd.read_csv('./calc/cgm/bergman_fit/all_fit_dataframe.csv')

# Extract subject information from the PATH
pattern = re.compile(r"dexcom_g6/([0-9]+)/")
subjects = [pattern.search(s).groups()[0] for s in paths[:bergman_df.shape[0]] if pattern.search(s)]






clinical_df = pd.read_csv('./data/aireadi/clinical_data/measurement.csv')

# Creatinine: 
# "measurement_source_value" = "Urine Creatinine (mg/dL)"
ur_creatinine_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "Urine Creatinine (mg/dL)"]
for i in range(tmp_df.shape[0]):
    ur_creatinine_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


creatinine_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "Creatinine (mg/dL)"]
for i in range(tmp_df.shape[0]):
    creatinine_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


insulin_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "INSULIN (ng/mL)"]
for i in range(tmp_df.shape[0]):
    insulin_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


glucose_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "Glucose (mg/dL)"]
for i in range(tmp_df.shape[0]):
    glucose_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]



agr_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "A/G Ratio"]
for i in range(tmp_df.shape[0]):
    agr_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


logmar_os_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "viaospscore, Photopic LogMAR OS Score"]
for i in range(tmp_df.shape[0]):
    logmar_os_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


logmar_od_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "viaodpscore, Photopic LogMAR OD Score"]
for i in range(tmp_df.shape[0]):
    logmar_od_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]

# HbA1c (%)
# "measurement_source_value" = "HbA1c (%)"
hgba1c_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "HbA1c (%)"]
for i in range(tmp_df.shape[0]):
    hgba1c_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]

######
# Capture clinical conditions in dataset
######

condition_df = pd.read_csv('./data/aireadi/clinical_data/condition_occurrence.csv')
diabetes_map = {}
tmp_df = condition_df[condition_df["condition_source_value"] == "mhterm_dm2, Type II Diabetes"]
for i in range(tmp_df.shape[0]):
    diabetes_map[str(tmp_df['person_id'].iloc[i])] = 'T2DM'


prediabetes_map = {}
tmp_df = condition_df[condition_df["condition_source_value"] == "mhterm_predm, Pre-diabetes"]
for i in range(tmp_df.shape[0]):
    prediabetes_map[str(tmp_df['person_id'].iloc[i])] = 'preDM'


full_diabetes_map = {}
for s in subjects:
    if s in diabetes_map:
        full_diabetes_map[s] = "T2DM"
    elif s in prediabetes_map:
        full_diabetes_map[s] = "PreDM"
    else:
        full_diabetes_map[s] = "Healthy"


dr_map = {}
tmp_df = condition_df[condition_df["condition_source_value"] == "mhoccur_pdr, Diabetic retinopathy (in one or both"]
for i in range(tmp_df.shape[0]):
    dr_map[str(tmp_df['person_id'].iloc[i])] = 'DR'

for s in subjects:
    if s not in dr_map:
        dr_map[s] = "Healthy"


kidney_map = {}
tmp_df = condition_df[condition_df["condition_source_value"] == "mhoccur_rnl, Kidney problems"]
for i in range(tmp_df.shape[0]):
    kidney_map[str(tmp_df['person_id'].iloc[i])] = 'KidneyDisease'

for s in subjects:
    if s not in kidney_map:
        kidney_map[s] = "Healthy"


glaucoma_map = {}
tmp_df = condition_df[condition_df["condition_source_value"] == "mhoccur_glc, Glaucoma (in one or both eyes)"]
for i in range(tmp_df.shape[0]):
    glaucoma_map[str(tmp_df['person_id'].iloc[i])] = 'Glaucoma'

for s in subjects:
    if s not in glaucoma_map:
        glaucoma_map[s] = "Healthy"



plot_df = pd.DataFrame.from_dict({
    "Subject": subjects,
    "UrineCr": [ur_creatinine_map[s] if s in ur_creatinine_map else np.nan for s in subjects],
    "SerumCr": [creatinine_map[s] if s in creatinine_map else np.nan for s in subjects],
    "Insulin": [insulin_map[s] if s in insulin_map else np.nan for s in subjects],
    "Glucose": [glucose_map[s] if s in glucose_map else np.nan for s in subjects],
    "AGRatio": [agr_map[s] if s in agr_map else np.nan for s in subjects],
    "LogMAROD": [logmar_od_map[s] if s in logmar_od_map else np.nan for s in subjects],
    "LogMAROS": [logmar_os_map[s] if s in logmar_os_map else np.nan for s in subjects],
    "HgbA1c": [hgba1c_map[s] if s in hgba1c_map else np.nan for s in subjects],
    "DMStatus": [full_diabetes_map[s] if s in full_diabetes_map else np.nan for s in subjects],
    "RetinopathyStatus": [dr_map[s] if s in dr_map else np.nan for s in subjects],
    "KidneyStatus": [kidney_map[s] if s in kidney_map else np.nan for s in subjects],
    "GlaucomaStatus": [glaucoma_map[s] if s in glaucoma_map else np.nan for s in subjects],
    "p1": bergman_df['p1'],
    "p2": bergman_df['p2'],
    "p3": bergman_df['p3']
})

plot_df['DMStatus'] = plot_df['DMStatus'].astype('category')
plot_df['RetinopathyStatus'] = plot_df['RetinopathyStatus'].astype('category')
plot_df['KidneyStatus'] = plot_df['KidneyStatus'].astype('category')
plot_df['GlaucomaStatus'] = plot_df['GlaucomaStatus'].astype('category')


plot_df["LogMARAvg"] = plot_df[["LogMAROD", "LogMAROS"]].mean(axis=1)

################################################################################
## Generate Plots
################################################################################

sc = plt.scatter(
    plot_df['p1'],
    plot_df['p2'],
    c=plot_df['HgbA1c'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("HgbA1c")

plt.xlabel("p1")
plt.ylabel("p2")
plt.title("MDS embedding colored by HgbA1c")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/hgba1c_p1_p2_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/hgba1c_p1_p2_plot.svg")            # vector SVG
plt.close()


sc = plt.scatter(
    plot_df['p1'],
    plot_df['p2'],
    c=plot_df['LogMARAvg'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("LogMARAvg")

plt.xlabel("p1")
plt.ylabel("p2")
plt.title("MDS embedding colored by LogMAR")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/logmar_p1_p2_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/logmar_p1_p2_plot.svg")            # vector SVG
plt.close()



sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['LogMARAvg'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("LogMARAvg")

plt.xlabel("p2")
plt.ylabel("p3")
plt.title("Colored by LogMAR")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/logmar_p2_p3_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/logmar_p2_p3_plot.svg")            # vector SVG
plt.close()



sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['Insulin'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("Insulin")

plt.xlabel("p2")
plt.ylabel("p3")
plt.title("Colored by Insulin")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/insulin_p2_p3_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/insulin_p2_p3_plot.svg")            # vector SVG
plt.close()



sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['AGRatio'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("AGRatio")

plt.xlabel("p2")
plt.ylabel("p3")
plt.title("Colored by AGRatio")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/agr_p2_p3_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/agr_p2_p3_plot.svg")            # vector SVG
plt.close()



sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['SerumCr'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("SerumCr")

plt.xlabel("p2")
plt.ylabel("p3")
plt.title("Colored by SerumCr")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/SerumCr_p2_p3_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/SerumCr_p2_p3_plot.svg")            # vector SVG
plt.close()


sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['Glucose'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("Glucose")

plt.xlabel("p2")
plt.ylabel("p3")
plt.title("Colored by Glucose")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/bergman_stats/Glucose_p2_p3_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/bergman_stats/Glucose_p2_p3_plot.svg")            # vector SVG
plt.close()

######
# Plot 
######

sc = plt.scatter(
    plot_df['p1'],
    plot_df['p2'],
    c=plot_df['DMStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['DMStatus'].cat.categories))
plt.show()

sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['DMStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['DMStatus'].cat.categories))
plt.show()



sc = plt.scatter(
    plot_df['p1'],
    plot_df['p2'],
    c=plot_df['RetinopathyStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['RetinopathyStatus'].cat.categories))
plt.show()

sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['RetinopathyStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['RetinopathyStatus'].cat.categories))
plt.show()



sc = plt.scatter(
    plot_df['p1'],
    plot_df['p2'],
    c=plot_df['KidneyStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['KidneyStatus'].cat.categories))
plt.show()

sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['KidneyStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['KidneyStatus'].cat.categories))
plt.show()

sc = plt.scatter(
    plot_df['p1'],
    plot_df['p3'],
    c=plot_df['KidneyStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['KidneyStatus'].cat.categories))
plt.show()

sc = plt.scatter(
    plot_df['p1'],
    plot_df['p2'],
    c=plot_df['GlaucomaStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['GlaucomaStatus'].cat.categories))
plt.show()

sc = plt.scatter(
    plot_df['p2'],
    plot_df['p3'],
    c=plot_df['GlaucomaStatus'].cat.codes
)
# Automatically generate a proper categorical legend
plt.legend(handles=sc.legend_elements()[0], labels=list(plot_df['GlaucomaStatus'].cat.categories))
plt.show()


################################################################################
## Run Statistics
################################################################################


group_a = plot_df[plot_df['KidneyStatus'] == 'KidneyDisease']['p2']
group_b = plot_df[plot_df['KidneyStatus'] == 'Healthy']['p2']

# 3. Run the Independent T-Test
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=True)



group_a = plot_df[plot_df['KidneyStatus'] == 'KidneyDisease']['p2']
group_b = plot_df[plot_df['KidneyStatus'] == 'Healthy']['p2']

# 3. Run the Independent T-Test
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=True)

mask = ~np.isnan(plot_df['SerumCr'])
stats.pearsonr(plot_df['SerumCr'][mask], plot_df['p2'][mask])


mask = ~np.isnan(plot_df['SerumCr'])
stats.spearmanr(plot_df['SerumCr'][mask], plot_df['p2'][mask])


mask = ~np.isnan(plot_df['UrineCr'])
stats.spearmanr(plot_df['UrineCr'][mask], plot_df['p2'][mask])


mask = ~np.isnan(plot_df['HgbA1c'])
stats.spearmanr(plot_df['HgbA1c'][mask], plot_df['p2'][mask])




