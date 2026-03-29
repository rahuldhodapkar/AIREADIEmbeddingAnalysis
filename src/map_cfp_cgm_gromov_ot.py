#!/usr/bin/env python
#
# Use Gromov Optimal Transport to build a map between the CGM domain and the
# CFP domain based on CFP embeddings and the CGM dynamic time warp (DTW)
# distance matrices. Of note, these distance matrices are not true metrics
# as they do not satisfy the triangle inequality.
#
# @author Rahul Dhodapkar
#

import pickle
import numpy as np
import pandas as pd
import ot
import ot.gromov
import os
import re
from collections import defaultdict
from scipy.linalg import svd
from scipy import stats
import matplotlib.pyplot as plt

################################################################################
## BUILD OUTPUT SCAFFOLDING
################################################################################

os.makedirs('./calc/ot/cgm_to_cfp', exist_ok=True)
os.makedirs('./fig/ot/cgm_to_cfp', exist_ok=True)


################################################################################
## HELPER FUNCTIONS
################################################################################


def get_image_paths_os_walk(directory):
    """ Takes a base directory and returns all image paths. """
    image_paths = []
    image_extensions = ('.dcm') 
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_file_paths_os_walk(directory):
    """ Takes a base directory and returns all image paths. """
    file_paths = []
    file_extensions = ('.json') 
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(file_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths




def compute_alignment_score(cfp_subjects, cgm_subjects, T):
    """
    Compute label alignment score based on transport matrix.
    
    Args:
        cfp_subjects: list of source labels (length N)
        cgm_subjects: list of target labels (length M)
        T: (N x M) transport matrix
    
    Returns:
        accuracy score
    """
    cfp_subjects = np.array(cfp_subjects)
    cgm_subjects = np.array(cgm_subjects)
    #
    # For each source, pick target with highest transport mass
    mapped_indices = np.argmax(T, axis=0)
    mapped_labels = cfp_subjects[mapped_indices]
    #
    # Compare labels
    return np.mean(cgm_subjects == mapped_labels)


def permutation_test(cfp_subjects, cgm_subjects, T, n_permutations=1000, random_state=42):
    """
    Perform permutation test to assess non-randomness of transport mapping.
    
    Returns:
        observed_score, null_distribution, p_value
    """
    rng = np.random.default_rng(random_state)
    #
    # Observed score
    observed_score = compute_alignment_score(cfp_subjects, cgm_subjects, T)
    #
    null_scores = []
    #
    for _ in range(n_permutations):
        # Shuffle target labels
        shuffled_cgm = rng.permutation(cgm_subjects)
        #
        score = compute_alignment_score(cfp_subjects, shuffled_cgm, T)
        null_scores.append(score)
    #
    null_scores = np.array(null_scores)
    #
    # p-value (right-tailed)
    p_value = np.mean(null_scores >= observed_score)
    return (observed_score, null_scores, p_value)
    

################################################################################
## LOAD DATA
################################################################################

# Read CFP Distance Matrix (Fine-tune on APTOS2019; VIT)
with open('./calc/finetune/vit_aptos/finetune_vit_embed_dist.pkl', 'rb') as file:
    C1 = pickle.load(file)


candidate_paths = get_image_paths_os_walk("./data/aireadi/retinal_photography/cfp/icare_eidon")
paths = [s for s in candidate_paths if re.compile(".*_uwf_central_").match(s)]
pattern = re.compile(r"icare_eidon/([0-9]+)/")
cfp_subjects = [pattern.search(s).groups()[0] for s in paths if pattern.search(s)]

# Read CGM Distance Matrix
with open('./calc/cgm/fft/cgm_dtw_dist.pkl', 'rb') as file:
    C2 = pickle.load(file)

paths = get_file_paths_os_walk('./data/aireadi/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6')
pattern = re.compile(r"dexcom_g6/([0-9]+)/")
cgm_subjects = [pattern.search(s).groups()[0] for s in paths if pattern.search(s)]

################################################################################
## Perform Gromov OT
################################################################################
#
# Because no information about the subject identity of each patient is provided
# when doing Gromov OT, we can compare the distance between paired observations
# with the distance between observations at random - to show that there is
# a natural shared structure between the two data domains.
#

# Generate a subset of each dataset that has a 1-1 mapping between domains
# for easier comparison

common_subjects = list(set(cfp_subjects) & set(cgm_subjects))

cfp_indices_to_select = []
cgm_indices_to_select = []
for subj in common_subjects:
    for i in range(len(cfp_subjects)):
        if cfp_subjects[i] == subj:
            cfp_indices_to_select.append(i)
            break
    for i in range(len(cgm_subjects)):
        if cgm_subjects[i] == subj:
            cgm_indices_to_select.append(i)
            break


cfp_indices_to_select = sorted(cfp_indices_to_select)
cgm_indices_to_select = sorted(cgm_indices_to_select)


C1_sub = np.take(np.take(C1, cfp_indices_to_select, axis=0), cfp_indices_to_select, axis=1)
C2_sub = np.take(np.take(C2, cgm_indices_to_select, axis=0), cgm_indices_to_select, axis=1)

# C1_sub and C2_sub are now co-registered in terms of subjects

# Normalize (recommended for stability)
C1_sub /= C1_sub.max()
C2_sub /= C2_sub.max()

n1 = C1_sub.shape[0]
n2 = C2_sub.shape[0]

# Uniform distributions over points
p = np.ones(n1) / n1
q = np.ones(n2) / n2


# Compute GW distance (squared) + transport plan
gw_dist2, log = ot.gromov.gromov_wasserstein2(
    C1_sub,
    C2_sub,
    p,
    q,
    loss_fun="square_loss",
    log=True,
)

T = log["T"]  # transport plan

print("GW distance^2:", gw_dist2)
print("Transport shape:", T.shape)

correct_map_distance = []
avg_map_distance = []

for i in range(T.shape[0]):
    v1 = np.zeros(T.shape[0])
    v1[i] = 1
    v2 = v1 @ T
    dists = v2 @ C2_sub
    correct_map_distance.append(dists[i])
    avg_map_distance.append(np.mean(dists))



paired_diff = [correct_map_distance[i] - avg_map_distance[i] for i in range(len(correct_map_distance))]



t_stat, p_value = stats.ttest_ind(correct_map_distance, avg_map_distance)

print("t =", t_stat)
print("p =", p_value)

u_stat, p_value = stats.mannwhitneyu(correct_map_distance, avg_map_distance, alternative='two-sided')

print("U statistic:", u_stat)
print("p-value:", p_value)



plt.figure()
plt.boxplot(
    [correct_map_distance, avg_map_distance, paired_diff],
    labels=["Correct Target", "Average Target", "Paired Difference"],
    showfliers=False)
plt.axhline(0, linestyle="--")
plt.show()



diff = avg_map_distance - correct_map_distance
plt.boxplot(diff)
plt.axhline(0, linestyle="--")
plt.title("Paired Differences (B - A)")
plt.show()


print("All done!")
