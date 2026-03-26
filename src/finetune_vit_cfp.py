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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import torch

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

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

################################################################################
## HELPER FUNCTIONS
################################################################################





################################################################################
## LOAD DATA
################################################################################

train_df = pd.read_csv('./data/aptos2019/train.csv')
train_df['label'] = train_df['diagnosis'] 
train_df['path'] = ['./data/aptos2019/train_images/train_images/{}.png'.format(f) for f in train_df['id_code']]

test_df = pd.read_csv('./data/aptos2019/test.csv')
test_df['label'] = test_df['diagnosis'] 
test_df['path'] = ['./data/aptos2019/test_images/test_images/{}.png'.format(f) for f in test_df['id_code']]

val_df = pd.read_csv('./data/aptos2019/valid.csv')
val_df['label'] = val_df['diagnosis'] 
val_df['path'] = ['./data/aptos2019/val_images/val_images/{}.png'.format(f) for f in val_df['id_code']]


################################################################################
## FINETUNE
################################################################################


# train_df, val_df, test_df must already exist
# columns:
#   path  -> full path to PNG
#   label -> class label

# If labels are already integers 0..N-1, this still works.
labels = sorted(train_df["label"].unique().tolist())
label2id = {str(label): i for i, label in enumerate(labels)}
id2label = {i: str(label) for label, i in label2id.items()}

train_labels = train_df["label"].astype(str).map(label2id).astype(np.int64).to_numpy()
val_labels = val_df["label"].astype(str).map(label2id).astype(np.int64).to_numpy()
test_labels = test_df["label"].astype(str).map(label2id).astype(np.int64).to_numpy()

class PathDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
    #
    def __len__(self):
        return len(self.paths)
    #
    def __getitem__(self, idx):
        return {
            "path": self.paths[idx],
            "label": int(self.labels[idx]),
        }

train_dataset = PathDataset(train_df["path"].astype(str).tolist(), train_labels)
val_dataset = PathDataset(val_df["path"].astype(str).tolist(), val_labels)
test_dataset = PathDataset(test_df["path"].astype(str).tolist(), test_labels)

processor = AutoImageProcessor.from_pretrained(MODEL_CKPT)

class Collator:
    def __init__(self, processor):
        self.processor = processor
    #
    def __call__(self, batch):
        images = []
        labels = []
        for item in batch:
            with Image.open(item["path"]) as img:
                images.append(img.convert("RGB"))
            labels.append(item["label"])
        #
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        labels = torch.tensor(labels, dtype=torch.long)
        #
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

data_collator = Collator(processor)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CKPT,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

use_fp16 = torch.cuda.is_available()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=20,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    fp16=use_fp16,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

logs = pd.DataFrame(trainer.state.log_history)
logs.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)

plt.figure(figsize=(10, 6))

train_logs = logs.dropna(subset=["loss"]) if "loss" in logs.columns else pd.DataFrame()
if not train_logs.empty:
    plt.plot(train_logs["step"], train_logs["loss"], label="train_loss")

val_loss_logs = logs.dropna(subset=["eval_loss"]) if "eval_loss" in logs.columns else pd.DataFrame()
if not val_loss_logs.empty:
    plt.plot(val_loss_logs["step"], val_loss_logs["eval_loss"], label="val_loss")

acc_logs = logs.dropna(subset=["eval_accuracy"]) if "eval_accuracy" in logs.columns else pd.DataFrame()
if not acc_logs.empty:
    plt.plot(acc_logs["step"], acc_logs["eval_accuracy"], label="val_accuracy")

plt.xlabel("step")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=200)
plt.close()


print("All done!")
