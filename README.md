# AI-READI Multimodal Embedding Alignment

## Overview

This project explores how **embeddings from computer vision models** trained on the **AI-READI dataset** relate to key **clinical biomarkers** and **metabolic signals**. Specifically, we analyze whether visual representations encode information associated with:

* Hemoglobin A1c (HbA1c)
* Serum creatinine
* Continuous Glucose Monitoring (CGM) data

By extracting embeddings from a range of modern vision architectures (e.g., CNNs, Vision Transformers, and foundation models), we compare these latent representations to structured clinical data and time-series glucose measurements.

The goal is to understand whether **image-derived features capture clinically meaningful physiological patterns**, and to quantify these relationships through cross-modal alignment and analysis.

## Approach (High-Level)

* **Vision Embeddings**: Extract feature representations from diverse computer vision models applied to AI-READI imaging data.
* **Clinical Representations**: Construct normalized features from lab values (HbA1c, creatinine) and derived metrics from CGM time-series.
* **Cross-Modal Comparison**: Evaluate relationships between visual and clinical representations using statistical, geometric, and predictive methods.

## Research Focus

* Do vision model embeddings reflect underlying metabolic health?
* Which model types best align with clinical biomarkers?
* Can visual features predict glycemic control or kidney function?

## Notes

This is a research-focused project intended to study **multimodal representation learning in healthcare**. It is not designed for clinical use.


Author: Rahul Dhodapkar