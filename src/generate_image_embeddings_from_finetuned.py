#!/usr/bin/env python
#
# Fine-tune the Google vision transformer on the external APTOS 2019 dataset
# published by the Aravind eye hospital of India. This dataset contains labels
# as follows and was downloaded from Kaggle.
#
#      https://www.kaggle.com/datasets/mariaherrerot/aptos2019
#
# Labels:
#
#   0 = no retinopathy
#   1 = mild NPDR
#   2 = moderate NPDR
#   3 = severe NPDR
#   4 = PDR
#
# As defined by expert human graders following the principle of the 
# International Clinical Diabetic Retinopathy Disease Severity Scale (ICDRSS)
# 
# @author Rahul Dhodapkar
#

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import numpy as np
import pydicom
from pydicom.pixels import apply_rescale, apply_voi_lut
import re
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

################################################################################
## BUILD OUTPUT SCAFFOLDING
################################################################################

os.makedirs('./calc/finetune/vit_aptos', exist_ok=True)
os.makedirs('./fig/finetune/vit_aptos', exist_ok=True)

################################################################################
## DEFINE CONSTANTS
################################################################################

MODEL_CKPT = "google/vit-base-patch16-224"
OUTPUT_DIR = "./calc/finetune/vit_aptos"
MODEL_DIR = "./calc/finetune/vit_aptos"

################################################################################
## HELPER FUNCTIONS
################################################################################

def dicom_to_pil(dicom_path: str | Path) -> Image.Image:
    ds = pydicom.dcmread(str(dicom_path))
    #
    # pydicom decodes and reshapes pixel data for you
    arr = ds.pixel_array.astype(np.float32)
    #
    # Apply modality LUT / rescale if present (CT/MR often need this)
    try:
        arr = apply_rescale(arr, ds).astype(np.float32)
    except Exception:
        pass
    #
    # Apply VOI LUT or windowing if present
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    #
    # Handle inverted grayscale
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    #
    # If multiframe, take the first frame by default
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    #
    # Convert to 8-bit for general vision models
    arr = np.squeeze(arr)
    arr = arr - np.min(arr)
    max_val = np.max(arr)
    if max_val > 0:
        arr = arr / max_val
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    #
    # Most HF vision models expect 3-channel RGB
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        img = Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        img = Image.fromarray(arr[..., :3], mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    #
    return img


def get_image_paths_os_walk(directory):
    """ Takes a base directory and returns all image paths. """
    image_paths = []
    image_extensions = ('.dcm') 
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths




class VisionEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        #
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
    #
    @torch.no_grad()
    def embed(self, image):
        """
        image: PIL.Image, numpy array, or anything accepted by the HF image processor
        returns: 1D numpy embedding
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        #
        emb = self._extract_embedding(outputs)
        emb = F.normalize(emb, p=2, dim=-1)  # good default for similarity search
        return emb.squeeze(0).cpu().numpy()
    #
    def _extract_embedding(self, outputs):
        # 1) Many vision models expose a pooled vector directly
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        #
        # 2) Transformer-style models: use CLS token if present
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            x = outputs.last_hidden_state
            #
            # (batch, seq_len, hidden)
            if x.ndim == 3:
                # CLS token
                if x.shape[1] > 1:
                    return x[:, 0, :]
                return x.mean(dim=1)
            #
            # (batch, channels, height, width) for CNN-like backbones
            if x.ndim == 4:
                return x.mean(dim=(2, 3))
        #
        # 3) Fallback: average the last hidden state if available
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            x = outputs.hidden_states[-1]
            if x.ndim == 3:
                return x.mean(dim=1)
            if x.ndim == 4:
                return x.mean(dim=(2, 3))
        #
        raise ValueError("Could not extract an embedding from this model's outputs.")


################################################################################
## LOAD DATA
################################################################################

candidate_paths = get_image_paths_os_walk("./data/aireadi/retinal_photography/cfp/icare_eidon")
paths = [s for s in candidate_paths if re.compile(".*_uwf_central_").match(s)]

model_name = "google/vit-base-patch16-224"
# model_name = "open-eye/RETFound_MAE"
embedder = VisionEmbedder(model_name) # ***NOTE*** model_name is not used.


################################################################################
## GENERATE EMBEDDINGS
################################################################################

all_embeddings = []

for p in tqdm(paths):
    image = dicom_to_pil(p)
    vec = embedder.embed(image)
    all_embeddings.append(
        vec
    )


Dx = squareform(pdist(all_embeddings, metric="euclidean"))

# D = your (n x n) distance matrix
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)

X = mds.fit_transform(Dx)

# Save distance matrix
with open('./calc/finetune/vit_aptos/finetune_vit_embed_dist.pkl', 'wb') as file:
    pickle.dump(Dx, file)

print("Saved distance matrix to file")

################################################################################
## COMPARE WITH DIABETES PARAMETERS
################################################################################

# Extract subject information from the PATH
pattern = re.compile(r"icare_eidon/([0-9]+)/")
subjects = [pattern.search(s).groups()[0] for s in paths if pattern.search(s)]


clinical_df = pd.read_csv('./data/aireadi/clinical_data/measurement.csv')

# Creatinine: 
# "measurement_source_value" = "Urine Creatinine (mg/dL)"
creatinine_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "Urine Creatinine (mg/dL)"]
for i in range(tmp_df.shape[0]):
    creatinine_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]

# HbA1c (%)
# "measurement_source_value" = "HbA1c (%)"
hgba1c_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "HbA1c (%)"]
for i in range(tmp_df.shape[0]):
    hgba1c_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


plot_df = pd.DataFrame.from_dict({
    "Subject": subjects,
    "MDS1": X[:, 0],
    "MDS2": X[:, 1],
    "UrineCr": [creatinine_map[s] if s in creatinine_map else np.nan for s in subjects],
    "HgbA1c": [hgba1c_map[s] if s in hgba1c_map else np.nan for s in subjects],
})

plot_df.to_csv('./calc/cfp/embeddings/mds_finetune_plot_df.csv')



sc = plt.scatter(
    plot_df['MDS1'],
    plot_df['MDS2'],
    c=plot_df['HgbA1c'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("HgbA1c")

plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS embedding colored by HgbA1c")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cfp/embeddings/hgba1c_finetune_cfp_mds_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cfp/embeddings/hgba1c_finetune_cfp_mds_plot.svg")            # vector SVG

plt.show()



sc = plt.scatter(
    plot_df['MDS1'],
    plot_df['MDS2'],
    c=plot_df['UrineCr'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("[UrineCr]")

plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS embedding colored by Creatinine")



plt.tight_layout()

print("All done!")


