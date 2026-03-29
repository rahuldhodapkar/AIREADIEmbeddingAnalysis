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


################################################################################
## BUILD OUTPUT SCAFFOLDING
################################################################################

os.makedirs('./calc/ot/cgm_to_cfp', exist_ok=True)
os.makedirs('./fig/ot/cgm_to_cfp', exist_ok=True)


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

# Optional: Normalize the distance matrices
# C1 /= C1.max()
# C2 /= C2.max()

# ----------------------------
# Classical MDS
# ----------------------------
def classical_mds(D, n_components=10):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    pos = evals > 1e-12
    evals = evals[pos]
    evecs = evecs[:, pos]

    k = min(n_components, len(evals))
    return evecs[:, :k] * np.sqrt(evals[:k])


# ----------------------------
# Procrustes alignment
# ----------------------------
def fit_orthogonal_procrustes(X_src, X_tgt):
    X_src = np.asarray(X_src, dtype=float)
    X_tgt = np.asarray(X_tgt, dtype=float)

    src_mean = X_src.mean(axis=0)
    tgt_mean = X_tgt.mean(axis=0)

    Xs = X_src - src_mean
    Xt = X_tgt - tgt_mean

    M = Xs.T @ Xt
    U, _, Vt = svd(M, full_matrices=False)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return {"R": R, "src_mean": src_mean, "tgt_mean": tgt_mean}


def transform_source(X_src, model):
    return (np.asarray(X_src) - model["src_mean"]) @ model["R"] + model["tgt_mean"]


# ----------------------------
# Label utilities
# ----------------------------
def make_label_index_map(labels):
    lab2idx = defaultdict(list)
    for i, lab in enumerate(labels):
        lab2idx[lab].append(i)
    return lab2idx


def split_labels_for_train_test(cfp_subjects, cgm_subjects, test_frac=0.3, random_state=0):
    """
    Split by labels, not by points.

    Train labels are used only for fitting.
    Test labels are held out entirely for evaluation.
    """
    rng = np.random.default_rng(random_state)

    src_map = make_label_index_map(cfp_subjects)
    tgt_map = make_label_index_map(cgm_subjects)

    common_labels = sorted(set(src_map).intersection(tgt_map))
    if len(common_labels) < 2:
        raise ValueError("Need at least 2 common labels to split train/test by label.")

    rng.shuffle(common_labels)

    n_test = max(1, int(round(test_frac * len(common_labels))))
    n_test = min(n_test, len(common_labels) - 1)

    test_labels = set(common_labels[:n_test])
    train_labels = set(common_labels[n_test:])

    return train_labels, test_labels, src_map, tgt_map


def make_pairs_from_labels(labels, src_map, tgt_map):
    """
    All same-label cross-product pairs for the given label set.
    """
    pairs = []
    for lab in labels:
        src_idx = src_map.get(lab, [])
        tgt_idx = tgt_map.get(lab, [])
        if len(src_idx) == 0 or len(tgt_idx) == 0:
            continue
        pairs.extend([(i, j) for i in src_idx for j in tgt_idx])
    return pairs


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_alignment(X_src_aligned, X_tgt, cfp_subjects, cgm_subjects, test_labels, src_map, tgt_map):
    """
    Evaluate retrieval on held-out labels only.
    For each test source point, ask whether the nearest held-out target point
    has the same label.
    """
    cfp_subjects = np.asarray(cfp_subjects)
    cgm_subjects = np.asarray(cgm_subjects)

    test_src_idx = np.array([i for lab in test_labels for i in src_map[lab]], dtype=int)
    test_tgt_idx = np.array([j for lab in test_labels for j in tgt_map[lab]], dtype=int)

    if len(test_src_idx) == 0 or len(test_tgt_idx) == 0:
        raise ValueError("No held-out test points available.")

    Y = X_tgt[test_tgt_idx]
    Y_lab = cgm_subjects[test_tgt_idx]

    top1_hits = []
    same_label_ranks = []

    for i in test_src_idx:
        lab = cfp_subjects[i]
        x = X_src_aligned[i]

        dists = np.linalg.norm(Y - x, axis=1)

        # nearest target overall
        nn = np.argmin(dists)
        top1_hits.append(Y_lab[nn] == lab)

        # rank of nearest same-label target
        same_mask = (Y_lab == lab)
        if np.any(same_mask):
            d_same = np.min(dists[same_mask])
            rank = 1 + np.sum(dists < d_same)
            same_label_ranks.append(rank)

    return {
        "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else np.nan,
        "mean_same_label_rank": float(np.mean(same_label_ranks)) if same_label_ranks else np.nan,
        "n_test_source": int(len(test_src_idx)),
        "n_test_target": int(len(test_tgt_idx)),
        "n_evaluated": int(len(top1_hits)),
    }


def permutation_test_top1(X_src_aligned, X_tgt, cfp_subjects, cgm_subjects, test_labels, src_map, tgt_map,
                          n_perm=2000, random_state=0):
    rng = np.random.default_rng(random_state)

    cfp_subjects = np.asarray(cfp_subjects)
    cgm_subjects = np.asarray(cgm_subjects)

    test_src_idx = np.array([i for lab in test_labels for i in src_map[lab]], dtype=int)
    test_tgt_idx = np.array([j for lab in test_labels for j in tgt_map[lab]], dtype=int)

    Y = X_tgt[test_tgt_idx]
    true_labels = cgm_subjects[test_tgt_idx]

    def score(labels):
        hits = []
        labels = np.asarray(labels)
        for i in test_src_idx:
            lab = cfp_subjects[i]
            x = X_src_aligned[i]
            dists = np.linalg.norm(Y - x, axis=1)
            nn = np.argmin(dists)
            hits.append(labels[nn] == lab)
        return float(np.mean(hits)) if hits else np.nan

    observed = score(true_labels)
    null_scores = np.empty(n_perm, dtype=float)

    for b in range(n_perm):
        null_scores[b] = score(rng.permutation(true_labels))

    p_value = (1 + np.sum(null_scores >= observed)) / (n_perm + 1)

    return {
        "observed_top1_accuracy": observed,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores, ddof=1)),
        "p_value": float(p_value),
        "null_scores": null_scores,
    }


# ----------------------------
# End-to-end pipeline
# ----------------------------
def align_and_test(
    C1,
    C2,
    cfp_subjects,
    cgm_subjects,
    n_components=10,
    test_frac=0.3,
    random_state=0,
    n_perm=2000,
):
    # Embed
    X1 = classical_mds(C1, n_components=n_components)
    X2 = classical_mds(C2, n_components=n_components)

    # Split by label
    train_labels, test_labels, src_map, tgt_map = split_labels_for_train_test(
        cfp_subjects, cgm_subjects, test_frac=test_frac, random_state=random_state
    )

    # Training pairs = all same-label cross pairs for train labels
    train_pairs = make_pairs_from_labels(train_labels, src_map, tgt_map)

    if len(train_pairs) < n_components:
        raise ValueError(
            f"Not enough training pairs to fit a {n_components}-D alignment. "
            f"Got {len(train_pairs)} pairs across {len(train_labels)} train labels."
        )

    src_train_idx = np.array([i for i, j in train_pairs], dtype=int)
    tgt_train_idx = np.array([j for i, j in train_pairs], dtype=int)

    # Fit and transform
    model = fit_orthogonal_procrustes(X1[src_train_idx], X2[tgt_train_idx])
    X1_aligned = transform_source(X1, model)

    # Evaluate on held-out labels
    eval_result = evaluate_alignment(
        X1_aligned, X2, cfp_subjects, cgm_subjects, test_labels, src_map, tgt_map
    )

    perm_result = permutation_test_top1(
        X1_aligned, X2, cfp_subjects, cgm_subjects, test_labels, src_map, tgt_map,
        n_perm=n_perm, random_state=random_state
    )

    return {
        "embedding_source": X1,
        "embedding_target": X2,
        "alignment_model": model,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_pairs": train_pairs,
        "X1_aligned": X1_aligned,
        "evaluation": eval_result,
        "permutation_test": perm_result,
    }


result = align_and_test(
    C1=C1,
    C2=C2,
    cfp_subjects=cfp_subjects,
    cgm_subjects=cgm_subjects,
    n_components=10,
    test_frac=0.2,
    random_state=0,
    n_perm=1000,
)

print(result["evaluation"])
print(result["permutation_test"])
