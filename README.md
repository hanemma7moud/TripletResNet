\# TripletResNet: mTBI Diagnosis from 3D CT Scans



This repository contains the official implementation of the paper:

\*\*"TripletResNet: A Deep Metric Learning Approach for mTBI Diagnosis from 3D CT"\*\* \*Published in: Australasian Joint Conference on Artificial Intelligence (AJCAI 2025)\* \[Springer Link](https://link.springer.com/chapter/10.1007/978-981-95-4972-6\_23) | \[ArXiv Link](https://arxiv.org/abs/2311.14197)



\## Overview

TripletResNet utilizes a two-stage training process:

1\.  \*\*Metric Learning:\*\* Uses Triplet Loss to learn a discriminative embedding space for 3D CT scans.

2\.  \*\*Classification:\*\* Freezes the learned feature extractor and trains a lightweight classifier head.



\## Requirements

Install dependencies via pip:

```bash

pip install -r requirements.txt

Data Preparation
Due to privacy restrictions, the dataset is not included. To use this code with your own data, prepare CSV files in the data/ directory with the following format:

File: data/train.csv (and valid.csv, test.csv)

Code snippet

scan,label
/path/to/patient1_ct.nii.gz,0
/path/to/patient2_ct.nii.gz,1
...
scan: Absolute path to the NIfTI 3D image.

label: 0 for Normal, 1 for mTBI.

Usage
Stage 1: Train Metric Learning (Feature Extractor)

Bash

python train_metric.py
This saves the trained trunk model to results/metric_learning/models/.

Stage 2: Train Classifier Update train_classifier.py to point to your best saved model from Stage 1:

Python

PRETRAINED_PATH = "results/metric_learning/models/trunk_best.pth"
Then run:

Bash

python train_classifier.py
Citation
If you use this code, please cite:

Code snippet

@inproceedings{ellethy2025tripletresnet,
  title={TripletResNet: A Deep Metric Learning Approach for mTBI Diagnosis from 3D CT},
  author={Ellethy, Hanem and Chandra, Shekhar S and Vegh, Viktor},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={295--308},
  year={2025},
  organization={Springer}
}
