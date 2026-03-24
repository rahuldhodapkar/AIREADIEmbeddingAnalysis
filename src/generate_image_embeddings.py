#!/usr/bin/env python
#
# Read all images and generate embeddings from CFP
#
# Initially, choose only ICARE_EIDON, UWF, central images
#
# @author Rahul Dhodapkar
#

import os
import torch
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pydicom
from pydicom.pixels import apply_rescale, apply_voi_lut
from PIL import Image
import re
import pandas as pd
from tqdm import tqdm

################################################################################
## BUILD OUTPUT SCAFFOLDING
################################################################################

os.makedirs('./calc/cfp/embeddings', exist_ok=True)

################################################################################
## DEFINE CONSTANTS
################################################################################



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
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
embedder = VisionEmbedder(model_name)

embeddings_to_return = []

for p in tqdm(paths):
    image = dicom_to_pil(p)
    vec = embedder.embed(image)
    embeddings_to_return.append(
        {"PATH": p} |
        {"EMBED_{}".format(i): vec[i] for i in range(len(vec))}
    )

# Example usage with a Hugging Face vision model

image = dicom_to_pil("./data/aireadi/retinal_photography/cfp/icare_eidon/1001/1001_eidon_uwf_central_cfp_r_1.2.826.0.1.3680043.8.641.1.20230809.2041.31942.dcm")
embedding = get_vit_embedding(image)



print(embedding.shape)  # (768,) for vit-base

################################################################################
## GENERATE EMBEDDINGS
################################################################################


print("All done!")
